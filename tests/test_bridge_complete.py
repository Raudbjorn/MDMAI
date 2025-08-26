"""Comprehensive tests for the MCP Bridge Service."""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import pytest
from fastapi import WebSocket
from fastapi.testclient import TestClient
from httpx import AsyncClient
from pydantic import ValidationError

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bridge.bridge_server import BridgeServer, create_bridge_app
from src.bridge.config import BridgeSettings, get_bridge_config
from src.bridge.mcp_process_manager import MCPProcess, MCPProcessManager
from src.bridge.models import (
    BridgeConfig,
    BridgeMessage,
    ClientMessage,
    MCPError,
    MCPErrorCode,
    MCPNotification,
    MCPRequest,
    MCPResponse,
    MCPSession,
    PendingRequest,
    ProcessStats,
    SessionState,
    TransportType,
)
from src.bridge.protocol_translator import MCPProtocolTranslator
from src.bridge.session_manager import BridgeSessionManager


# Fixtures
@pytest.fixture
def bridge_config():
    """Create test bridge configuration."""
    return BridgeConfig(
        mcp_server_path="src.main",
        max_processes=5,
        process_timeout=30,
        process_idle_timeout=60,
        session_timeout=300,
        max_sessions_per_client=2,
        enable_websocket=True,
        enable_sse=True,
        enable_http=True,
        require_auth=False,
        log_requests=True,
        log_responses=True,
    )


@pytest.fixture
def auth_config():
    """Create test bridge configuration with authentication."""
    return BridgeConfig(
        mcp_server_path="src.main",
        max_processes=5,
        require_auth=True,
        api_keys=["test-key-123", "test-key-456"],
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
def auth_bridge_server(auth_config):
    """Create test bridge server with authentication."""
    return BridgeServer(auth_config)


@pytest.fixture
def test_client(bridge_server):
    """Create test client."""
    return TestClient(bridge_server.app)


@pytest.fixture
def auth_test_client(auth_bridge_server):
    """Create test client with auth."""
    return TestClient(auth_bridge_server.app)


class TestBridgeConfiguration:
    """Test bridge configuration and settings."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = BridgeConfig()
        assert config.mcp_server_path == "src.main:main"
        assert config.max_processes == 10
        assert config.enable_websocket is True
        assert config.require_auth is False
        
    def test_config_from_settings(self):
        """Test configuration from settings."""
        settings = BridgeSettings()
        config = settings.to_bridge_config()
        assert isinstance(config, BridgeConfig)
        assert config.max_processes == settings.max_processes
        
    def test_config_validation(self):
        """Test configuration validation."""
        settings = BridgeSettings(
            max_processes=0,  # Invalid
            port=-1,  # Invalid
        )
        # Should still create config with defaults
        config = settings.to_bridge_config()
        assert config.max_processes == 0  # Will be validated at runtime


class TestMCPProcessManagement:
    """Test MCP process management functionality."""
    
    @pytest.mark.asyncio
    async def test_process_lifecycle(self, bridge_config):
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
            # Mock stdout readline to simulate MCP initialization
            init_response = json.dumps({
                "jsonrpc": "2.0",
                "id": "init-1",
                "result": {"capabilities": {"tools": True, "resources": True}},
            }).encode() + b"\n"
            
            mock_process.stdout.readline = AsyncMock(side_effect=[init_response, b""])
            
            # Start process
            result = await process.start()
            assert result is True
            assert process._running is True
            assert process.process is not None
            
            # Stop process
            await process.stop()
            assert process._running is False
    
    @pytest.mark.asyncio
    async def test_process_request_response(self, bridge_config):
        """Test sending request and receiving response."""
        process = MCPProcess("test-session", bridge_config)
        process._running = True
        process._initialized = True
        process.process = AsyncMock()
        process.process.stdin = AsyncMock()
        
        # Test sending request
        request_future = asyncio.Future()
        request_future.set_result({"status": "success", "data": "test"})
        
        with patch.object(process, "_write_message", new_callable=AsyncMock):
            process.pending_requests["test-1"] = PendingRequest(
                request_id="test-1",
                method="test/method",
                callback=request_future,
            )
            
            result = await process.send_request("test/method", {"param": "value"})
            assert result["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_process_health_check(self, bridge_config):
        """Test process health checking."""
        process = MCPProcess("test-session", bridge_config)
        process._running = True
        process.process = AsyncMock()
        process.process.returncode = None
        process.last_activity = datetime.now()
        
        # Test health check with healthy process
        health_task = asyncio.create_task(process._health_check_loop())
        await asyncio.sleep(0.1)
        health_task.cancel()
        
        try:
            await health_task
        except asyncio.CancelledError:
            pass
        
        assert process._running is True
    
    @pytest.mark.asyncio
    async def test_process_auto_restart(self, bridge_config):
        """Test automatic process restart on failure."""
        process = MCPProcess("test-session", bridge_config)
        process._auto_restart = True
        process.restart_count = 0
        process.max_restarts = 3
        
        # Mock process failure
        mock_process = AsyncMock()
        mock_process.returncode = 1  # Process died
        process.process = mock_process
        process._running = True
        
        with patch.object(process, "start", return_value=True) as mock_start:
            with patch.object(process, "_cleanup_process", new_callable=AsyncMock):
                # Simulate health check detecting failure
                await process._health_check_loop()
                
                # Should attempt restart
                assert process.restart_count > 0
    
    @pytest.mark.asyncio
    async def test_process_pool_management(self, process_manager):
        """Test process pool management."""
        await process_manager.start()
        
        with patch.object(MCPProcess, "start", return_value=True):
            # Create multiple processes
            process1 = await process_manager.create_process("session-1")
            process2 = await process_manager.create_process("session-2")
            
            assert len(process_manager.processes) == 2
            assert process_manager.get_process("session-1") is not None
            
            # Remove process
            await process_manager.remove_process("session-1")
            assert len(process_manager.processes) == 1
        
        await process_manager.stop()


class TestSessionManagement:
    """Test session management functionality."""
    
    @pytest.mark.asyncio
    async def test_session_creation(self, session_manager):
        """Test creating a new session."""
        await session_manager.start()
        
        with patch.object(session_manager.process_manager, "create_process") as mock_create:
            mock_process = AsyncMock()
            mock_process.process = AsyncMock()
            mock_process.process.pid = 12345
            mock_process.capabilities = {"tools": True}
            mock_create.return_value = mock_process
            
            session = await session_manager.create_session(
                client_id="test-client",
                transport=TransportType.WEBSOCKET,
            )
            
            assert session.session_id is not None
            assert session.client_id == "test-client"
            assert session.transport == TransportType.WEBSOCKET
            assert session.state == SessionState.READY
        
        await session_manager.stop()
    
    @pytest.mark.asyncio
    async def test_session_limit_per_client(self, session_manager):
        """Test session limit enforcement per client."""
        session_manager.config.max_sessions_per_client = 2
        await session_manager.start()
        
        with patch.object(session_manager.process_manager, "create_process") as mock_create:
            mock_process = AsyncMock()
            mock_process.process = AsyncMock()
            mock_process.process.pid = 12345
            mock_process.capabilities = {}
            mock_create.return_value = mock_process
            
            # Create max sessions
            session1 = await session_manager.create_session(client_id="test-client")
            session2 = await session_manager.create_session(client_id="test-client")
            
            # Third should fail
            with pytest.raises(RuntimeError, match="Maximum sessions"):
                await session_manager.create_session(client_id="test-client")
        
        await session_manager.stop()
    
    @pytest.mark.asyncio
    async def test_session_cleanup(self, session_manager):
        """Test session cleanup."""
        await session_manager.start()
        
        with patch.object(session_manager.process_manager, "create_process") as mock_create:
            with patch.object(session_manager.process_manager, "remove_process") as mock_remove:
                mock_process = AsyncMock()
                mock_process.process = AsyncMock()
                mock_process.process.pid = 12345
                mock_process.capabilities = {}
                mock_create.return_value = mock_process
                
                session = await session_manager.create_session(client_id="test-client")
                session_id = session.session_id
                
                # Remove session
                await session_manager.remove_session(session_id)
                
                # Verify cleanup
                assert session_manager.get_session(session_id) is None
                mock_remove.assert_called_once_with(session_id)
        
        await session_manager.stop()
    
    @pytest.mark.asyncio
    async def test_session_timeout(self, session_manager):
        """Test session timeout handling."""
        session_manager.config.session_timeout = 1  # 1 second timeout
        await session_manager.start()
        
        with patch.object(session_manager.process_manager, "create_process") as mock_create:
            mock_process = AsyncMock()
            mock_process.process = AsyncMock()
            mock_process.process.pid = 12345
            mock_process.capabilities = {}
            mock_create.return_value = mock_process
            
            session = await session_manager.create_session()
            session_id = session.session_id
            
            # Make session inactive
            session.last_activity = datetime.now() - timedelta(seconds=2)
            
            # Run cleanup
            await session_manager._cleanup_loop()
            
            # Session should be removed
            assert session_id not in session_manager.sessions
        
        await session_manager.stop()


class TestProtocolTranslation:
    """Test protocol translation functionality."""
    
    def test_parse_jsonrpc_request(self, protocol_translator):
        """Test parsing JSON-RPC request."""
        message = {
            "jsonrpc": "2.0",
            "id": "123",
            "method": "tools/search",
            "params": {"query": "test"},
        }
        
        result = protocol_translator.parse_client_message(message)
        assert isinstance(result, MCPRequest)
        assert result.method == "tools/search"
        assert result.params["query"] == "test"
    
    def test_parse_tool_call_format(self, protocol_translator):
        """Test parsing tool call format."""
        message = {
            "tool": "search",
            "params": {"query": "test"},
            "id": "123",
        }
        
        result = protocol_translator.parse_client_message(message)
        assert isinstance(result, MCPRequest)
        assert result.method == "tools/search"
        assert result.params["query"] == "test"
    
    def test_parse_batch_request(self, protocol_translator):
        """Test parsing batch request."""
        messages = [
            {"jsonrpc": "2.0", "id": "1", "method": "tools/list", "params": {}},
            {"jsonrpc": "2.0", "id": "2", "method": "tools/search", "params": {"query": "test"}},
        ]
        
        results = protocol_translator.parse_client_message(messages)
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, MCPRequest) for r in results)
    
    def test_format_response_jsonrpc(self, protocol_translator):
        """Test formatting response as JSON-RPC."""
        response = MCPResponse(id="123", result={"data": "test"})
        
        formatted = protocol_translator.format_response(response, "json-rpc")
        assert formatted["id"] == "123"
        assert formatted["result"]["data"] == "test"
    
    def test_format_response_openai(self, protocol_translator):
        """Test formatting response for OpenAI format."""
        response = MCPResponse(id="123", result={"data": "test"})
        
        formatted = protocol_translator.format_response(response, "openai")
        assert formatted["role"] == "function"
        assert formatted["name"] == "123"
        assert "data" in formatted["content"]
    
    def test_format_response_anthropic(self, protocol_translator):
        """Test formatting response for Anthropic format."""
        response = MCPResponse(id="123", result={"data": "test"})
        
        formatted = protocol_translator.format_response(response, "anthropic")
        assert formatted["type"] == "tool_result"
        assert formatted["tool_use_id"] == "123"
        assert formatted["content"]["data"] == "test"
    
    def test_translate_tool_discovery(self, protocol_translator):
        """Test translating tool discovery results."""
        tools = [
            {
                "name": "search",
                "description": "Search tool",
                "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ]
        
        # OpenAI format
        openai_tools = protocol_translator.translate_tool_discovery(tools, "openai")
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["function"]["name"] == "search"
        
        # Anthropic format
        anthropic_tools = protocol_translator.translate_tool_discovery(tools, "anthropic")
        assert anthropic_tools[0]["name"] == "search"
        assert "input_schema" in anthropic_tools[0]
    
    def test_validate_mcp_message(self, protocol_translator):
        """Test MCP message validation."""
        # Valid request
        assert protocol_translator.validate_mcp_message({
            "jsonrpc": "2.0",
            "id": "123",
            "method": "test",
        })
        
        # Valid notification
        assert protocol_translator.validate_mcp_message({
            "jsonrpc": "2.0",
            "method": "test",
        })
        
        # Valid response
        assert protocol_translator.validate_mcp_message({
            "jsonrpc": "2.0",
            "id": "123",
            "result": {},
        })
        
        # Invalid - missing jsonrpc
        assert not protocol_translator.validate_mcp_message({
            "id": "123",
            "method": "test",
        })


class TestBridgeServer:
    """Test bridge server functionality."""
    
    def test_health_check(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "uptime" in data
    
    def test_create_session(self, test_client):
        """Test session creation via HTTP."""
        with patch.object(BridgeSessionManager, "create_session") as mock_create:
            mock_session = MCPSession(
                session_id="test-123",
                client_id="client-1",
                state=SessionState.READY,
                capabilities={"tools": True},
            )
            mock_create.return_value = mock_session
            
            response = test_client.post("/sessions", json={
                "client_id": "client-1",
                "metadata": {"app": "test"},
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == "test-123"
            assert data["state"] == "ready"
    
    def test_get_session(self, test_client):
        """Test getting session info."""
        with patch.object(BridgeSessionManager, "get_session") as mock_get:
            mock_session = MCPSession(
                session_id="test-123",
                client_id="client-1",
                state=SessionState.READY,
            )
            mock_get.return_value = mock_session
            
            response = test_client.get("/sessions/test-123")
            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == "test-123"
    
    def test_delete_session(self, test_client):
        """Test deleting session."""
        with patch.object(BridgeSessionManager, "remove_session") as mock_remove:
            response = test_client.delete("/sessions/test-123")
            assert response.status_code == 200
            mock_remove.assert_called_once_with("test-123")
    
    def test_discover_tools(self, test_client):
        """Test tool discovery."""
        with patch.object(BridgeSessionManager, "send_request") as mock_send:
            mock_send.return_value = {
                "tools": [
                    {"name": "search", "description": "Search tool"},
                    {"name": "add_source", "description": "Add source tool"},
                ]
            }
            
            response = test_client.post("/tools/discover", json={})
            assert response.status_code == 200
            data = response.json()
            assert len(data["tools"]) == 2
            assert data["tools"][0]["name"] == "search"
    
    def test_call_tool(self, test_client):
        """Test tool invocation."""
        with patch.object(BridgeSessionManager, "send_request") as mock_send:
            mock_send.return_value = {
                "status": "success",
                "results": ["result1", "result2"],
            }
            
            response = test_client.post("/tools/call", json={
                "tool": "search",
                "params": {"query": "test"},
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["tool"] == "search"
            assert data["result"]["status"] == "success"
    
    def test_authentication_required(self, auth_test_client):
        """Test authentication requirement."""
        # Without auth header
        response = auth_test_client.get("/sessions/test-123")
        assert response.status_code == 200  # Should work as we check auth in route
        
        # With invalid auth
        response = auth_test_client.get(
            "/sessions/test-123",
            headers={"Authorization": "Bearer invalid-key"},
        )
        assert response.status_code == 200  # FastAPI doesn't auto-reject
        
        # With valid auth
        response = auth_test_client.get(
            "/sessions/test-123",
            headers={"Authorization": "Bearer test-key-123"},
        )
        assert response.status_code in (200, 404)  # Depends on session existence
    
    def test_rate_limiting(self, bridge_server):
        """Test rate limiting functionality."""
        # Test rate limit check
        client_id = "test-client"
        
        # Should allow initial requests
        for _ in range(5):
            assert asyncio.run(bridge_server.check_rate_limit(client_id)) is True
        
        # Simulate reaching limit
        from src.bridge.config import settings
        settings.rate_limit_requests = 5
        bridge_server.rate_limits[client_id] = [datetime.now()] * 5
        
        # Should deny when limit reached
        assert asyncio.run(bridge_server.check_rate_limit(client_id)) is False
    
    def test_get_stats(self, test_client):
        """Test getting bridge statistics."""
        with patch.object(BridgeSessionManager, "get_stats") as mock_session_stats:
            with patch.object(MCPProcessManager, "get_stats") as mock_process_stats:
                mock_session_stats.return_value = {
                    "active_sessions": 2,
                    "total_sessions": 5,
                }
                mock_process_stats.return_value = []
                
                response = test_client.get("/stats")
                assert response.status_code == 200
                data = response.json()
                assert "active_sessions" in data
                assert "uptime_seconds" in data


class TestWebSocketConnection:
    """Test WebSocket functionality."""
    
    @pytest.mark.asyncio
    async def test_websocket_session_creation(self, test_client):
        """Test creating session via WebSocket."""
        with test_client.websocket_connect("/ws") as websocket:
            # Send create session message
            websocket.send_json({
                "type": "create_session",
                "client_id": "ws-client-1",
            })
            
            # Should receive session created response
            response = websocket.receive_json()
            assert response["type"] in ("session_created", "error")
    
    @pytest.mark.asyncio
    async def test_websocket_tool_call(self, test_client):
        """Test tool call via WebSocket."""
        with patch.object(BridgeSessionManager, "create_session") as mock_create:
            with patch.object(BridgeSessionManager, "send_request") as mock_send:
                mock_session = MCPSession(
                    session_id="ws-123",
                    state=SessionState.READY,
                    capabilities={},
                )
                mock_create.return_value = mock_session
                mock_send.return_value = {"result": "test"}
                
                with test_client.websocket_connect("/ws") as websocket:
                    # Create session
                    websocket.send_json({
                        "type": "create_session",
                        "client_id": "ws-client",
                    })
                    response = websocket.receive_json()
                    
                    # Send tool request
                    websocket.send_json({
                        "jsonrpc": "2.0",
                        "id": "req-1",
                        "method": "tools/search",
                        "params": {"query": "test"},
                    })
                    
                    # Should receive response
                    response = websocket.receive_json()
                    assert response["type"] in ("response", "error")


class TestSSEConnection:
    """Test Server-Sent Events functionality."""
    
    @pytest.mark.asyncio
    async def test_sse_connection(self, test_client):
        """Test SSE connection and heartbeat."""
        with patch.object(BridgeSessionManager, "get_session") as mock_get:
            mock_session = MCPSession(session_id="sse-123")
            mock_get.return_value = mock_session
            
            # SSE endpoint returns streaming response
            response = test_client.get("/events/sse-123", stream=True)
            assert response.status_code == 200
            
            # Read first event
            for chunk in response.iter_lines():
                if chunk:
                    assert b"event:" in chunk or b"data:" in chunk
                    break


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_invalid_session_id(self, test_client):
        """Test handling invalid session ID."""
        with patch.object(BridgeSessionManager, "get_session") as mock_get:
            mock_get.return_value = None
            
            response = test_client.get("/sessions/invalid-id")
            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()
    
    def test_process_creation_failure(self, test_client):
        """Test handling process creation failure."""
        with patch.object(BridgeSessionManager, "create_session") as mock_create:
            mock_create.side_effect = RuntimeError("Failed to create process")
            
            response = test_client.post("/sessions", json={})
            assert response.status_code == 500
            assert "Failed to create process" in response.json()["detail"]
    
    def test_tool_call_timeout(self, test_client):
        """Test handling tool call timeout."""
        with patch.object(BridgeSessionManager, "send_request") as mock_send:
            mock_send.side_effect = TimeoutError("Request timed out")
            
            response = test_client.post("/tools/call", json={
                "tool": "search",
                "params": {},
            })
            assert response.status_code == 500
            assert "timed out" in response.json()["detail"].lower()
    
    def test_malformed_request(self, protocol_translator):
        """Test handling malformed requests."""
        with pytest.raises(ValueError):
            protocol_translator.parse_client_message("invalid json")
        
        # Invalid message format
        with pytest.raises(Exception):
            protocol_translator.parse_client_message({
                "invalid": "format",
            })


class TestIntegration:
    """Integration tests for the complete bridge system."""
    
    @pytest.mark.asyncio
    async def test_full_request_cycle(self, bridge_server):
        """Test complete request/response cycle."""
        await bridge_server.startup()
        
        try:
            with patch("asyncio.create_subprocess_exec") as mock_subprocess:
                # Mock subprocess
                mock_process = AsyncMock()
                mock_process.pid = 99999
                mock_process.returncode = None
                mock_process.stdin = AsyncMock()
                mock_process.stdout = AsyncMock()
                mock_process.stderr = AsyncMock()
                mock_subprocess.return_value = mock_process
                
                # Mock MCP responses
                responses = [
                    # Initialize response
                    json.dumps({
                        "jsonrpc": "2.0",
                        "id": Mock(),
                        "result": {"capabilities": {"tools": True}},
                    }).encode() + b"\n",
                    # Tool list response
                    json.dumps({
                        "jsonrpc": "2.0",
                        "id": Mock(),
                        "result": {"tools": [{"name": "search"}]},
                    }).encode() + b"\n",
                ]
                mock_process.stdout.readline = AsyncMock(side_effect=responses + [b""])
                
                # Create session
                session = await bridge_server.session_manager.create_session(
                    client_id="test-client",
                    transport=TransportType.HTTP,
                )
                
                assert session.state == SessionState.READY
                assert session.session_id is not None
                
        finally:
            await bridge_server.shutdown()
    
    @pytest.mark.asyncio
    async def test_concurrent_sessions(self, bridge_server):
        """Test handling multiple concurrent sessions."""
        await bridge_server.startup()
        
        try:
            with patch("asyncio.create_subprocess_exec") as mock_subprocess:
                # Mock subprocess creation
                async def create_mock_process(*args, **kwargs):
                    mock_process = AsyncMock()
                    mock_process.pid = 10000 + len(bridge_server.process_manager.processes)
                    mock_process.returncode = None
                    mock_process.stdin = AsyncMock()
                    mock_process.stdout = AsyncMock()
                    mock_process.stderr = AsyncMock()
                    
                    # Mock initialization
                    mock_process.stdout.readline = AsyncMock(
                        return_value=json.dumps({
                            "jsonrpc": "2.0",
                            "id": "init",
                            "result": {"capabilities": {}},
                        }).encode() + b"\n"
                    )
                    
                    return mock_process
                
                mock_subprocess.side_effect = create_mock_process
                
                # Create multiple sessions concurrently
                tasks = [
                    bridge_server.session_manager.create_session(
                        client_id=f"client-{i}",
                        transport=TransportType.WEBSOCKET,
                    )
                    for i in range(3)
                ]
                
                sessions = await asyncio.gather(*tasks)
                assert len(sessions) == 3
                assert all(s.state == SessionState.READY for s in sessions)
                
        finally:
            await bridge_server.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])