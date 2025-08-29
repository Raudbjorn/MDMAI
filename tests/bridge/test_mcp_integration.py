"""Integration tests for MCP communication and protocol handling."""

import asyncio
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from pydantic import ValidationError

from src.bridge.mcp_process_manager import MCPProcess, MCPProcessManager
from src.bridge.models import (
    BridgeConfig,
    MCPError,
    MCPErrorCode,
    MCPNotification,
    MCPRequest,
    MCPResponse,
    MCPSession,
    PendingRequest,
    SessionState,
)
from src.bridge.protocol_translator import MCPProtocolTranslator
from src.bridge.session_manager import BridgeSessionManager


@pytest.fixture
def integration_config() -> BridgeConfig:
    """Create configuration for integration testing."""
    return BridgeConfig(
        mcp_server_path="python",  # Use Python as a mock MCP server
        mcp_server_args=["-m", "src.main"],
        process_timeout=30,
        session_timeout=60,
        max_processes=5,
        max_sessions_per_client=3,
        enable_process_pool=True,
        process_pool_size=2,
    )


@pytest.fixture
def mock_mcp_script() -> str:
    """Create a mock MCP server script for testing."""
    script = '''
import sys
import json
import time

def handle_request(request):
    """Handle incoming MCP request."""
    method = request.get("method", "")
    params = request.get("params", {})
    
    if method == "initialize":
        return {
            "capabilities": {
                "tools": ["search", "analyze", "calculate"],
                "prompts": True,
                "resources": True,
            },
            "serverInfo": {
                "name": "test-mcp-server",
                "version": "1.0.0",
            }
        }
    elif method == "tools/list":
        return {
            "tools": [
                {
                    "name": "search",
                    "description": "Search for information",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "analyze",
                    "description": "Analyze data",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "data": {"type": "array"}
                        }
                    }
                }
            ]
        }
    elif method == "tools/call":
        tool_name = params.get("name", "")
        if tool_name == "search":
            query = params.get("arguments", {}).get("query", "")
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Search results for: {query}"
                    }
                ]
            }
        elif tool_name == "analyze":
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "Analysis complete"
                    }
                ]
            }
    elif method == "prompts/list":
        return {
            "prompts": [
                {
                    "name": "summarize",
                    "description": "Summarize text",
                    "arguments": [
                        {
                            "name": "text",
                            "description": "Text to summarize",
                            "required": True
                        }
                    ]
                }
            ]
        }
    elif method == "resources/list":
        return {
            "resources": [
                {
                    "uri": "file://test.txt",
                    "name": "test.txt",
                    "mimeType": "text/plain"
                }
            ]
        }
    
    return {"error": {"code": -32601, "message": "Method not found"}}

# Main loop
while True:
    try:
        line = sys.stdin.readline()
        if not line:
            break
        
        request = json.loads(line.strip())
        response = {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": handle_request(request)
        }
        
        print(json.dumps(response))
        sys.stdout.flush()
        
    except json.JSONDecodeError:
        error_response = {
            "jsonrpc": "2.0",
            "error": {
                "code": -32700,
                "message": "Parse error"
            }
        }
        print(json.dumps(error_response))
        sys.stdout.flush()
    except Exception as e:
        error_response = {
            "jsonrpc": "2.0",
            "error": {
                "code": -32603,
                "message": str(e)
            }
        }
        print(json.dumps(error_response))
        sys.stdout.flush()
'''
    return script


class TestMCPProcessIntegration:
    """Test MCP process lifecycle and communication."""
    
    @pytest.mark.asyncio
    async def test_process_initialization(self, integration_config: BridgeConfig):
        """Test MCP process initialization and capability negotiation."""
        process = MCPProcess("test-session", integration_config)
        
        # Mock subprocess creation
        mock_process = AsyncMock()
        mock_process.pid = 12345
        mock_process.returncode = None
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        
        # Mock initialization response
        init_response = json.dumps({
            "jsonrpc": "2.0",
            "id": "init-1",
            "result": {
                "capabilities": {
                    "tools": ["search", "analyze"],
                    "prompts": True,
                },
                "serverInfo": {
                    "name": "test-server",
                    "version": "1.0.0",
                }
            }
        }).encode() + b"\n"
        
        mock_process.stdout.readline = AsyncMock(side_effect=[
            init_response,
            b"",  # EOF
        ])
        
        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            # Start process
            result = await process.start()
            
            assert result is True
            assert process._running is True
            assert process._initialized is True
            assert process.capabilities == {
                "tools": ["search", "analyze"],
                "prompts": True,
            }
            assert process.server_info["name"] == "test-server"
    
    @pytest.mark.asyncio
    async def test_process_request_response(self, integration_config: BridgeConfig):
        """Test sending requests and receiving responses from MCP process."""
        process = MCPProcess("test-session", integration_config)
        process._running = True
        process._initialized = True
        
        # Setup mock process
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()
        process.process = mock_process
        
        # Setup response handling
        response_data = {
            "jsonrpc": "2.0",
            "id": "req-123",
            "result": {"status": "success", "data": [1, 2, 3]}
        }
        
        async def mock_readline():
            return json.dumps(response_data).encode() + b"\n"
        
        mock_process.stdout.readline = mock_readline
        
        # Start read loop
        read_task = asyncio.create_task(process._read_output())
        
        # Send request
        with patch.object(process, "_generate_id", return_value="req-123"):
            result = await process.send_request(
                "test_method",
                {"param1": "value1"}
            )
        
        # Cancel read task
        read_task.cancel()
        try:
            await read_task
        except asyncio.CancelledError:
            pass
        
        assert result == {"status": "success", "data": [1, 2, 3]}
    
    @pytest.mark.asyncio
    async def test_process_error_handling(self, integration_config: BridgeConfig):
        """Test MCP process error response handling."""
        process = MCPProcess("test-session", integration_config)
        process._running = True
        process._initialized = True
        
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()
        process.process = mock_process
        
        # Setup error response
        error_response = {
            "jsonrpc": "2.0",
            "id": "req-456",
            "error": {
                "code": MCPErrorCode.METHOD_NOT_FOUND,
                "message": "Unknown method",
                "data": {"method": "invalid_method"}
            }
        }
        
        async def mock_readline():
            return json.dumps(error_response).encode() + b"\n"
        
        mock_process.stdout.readline = mock_readline
        
        # Start read loop
        read_task = asyncio.create_task(process._read_output())
        
        # Send request that will receive error
        with patch.object(process, "_generate_id", return_value="req-456"):
            with pytest.raises(RuntimeError, match="Unknown method"):
                await process.send_request("invalid_method", {})
        
        # Cancel read task
        read_task.cancel()
        try:
            await read_task
        except asyncio.CancelledError:
            pass
    
    @pytest.mark.asyncio
    async def test_process_notification_handling(self, integration_config: BridgeConfig):
        """Test handling of MCP notifications from process."""
        process = MCPProcess("test-session", integration_config)
        process._running = True
        process._initialized = True
        
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()
        process.process = mock_process
        
        notifications_received = []
        
        # Setup notification handler
        def notification_handler(notification: MCPNotification):
            notifications_received.append(notification)
        
        process.notification_handler = notification_handler
        
        # Setup notification message
        notification = {
            "jsonrpc": "2.0",
            "method": "progress",
            "params": {"percentage": 50, "message": "Processing..."}
        }
        
        responses = [
            json.dumps(notification).encode() + b"\n",
            b"",  # EOF
        ]
        
        mock_process.stdout.readline = AsyncMock(side_effect=responses)
        
        # Run read loop
        await process._read_output()
        
        assert len(notifications_received) == 1
        assert notifications_received[0].method == "progress"
        assert notifications_received[0].params["percentage"] == 50
    
    @pytest.mark.asyncio
    async def test_process_timeout_handling(self, integration_config: BridgeConfig):
        """Test request timeout handling."""
        integration_config.process_timeout = 1  # 1 second timeout
        process = MCPProcess("test-session", integration_config)
        process._running = True
        process._initialized = True
        
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()
        process.process = mock_process
        
        # Mock readline that never returns (simulating timeout)
        async def mock_readline():
            await asyncio.sleep(10)  # Longer than timeout
            return b""
        
        mock_process.stdout.readline = mock_readline
        
        # Start read loop
        read_task = asyncio.create_task(process._read_output())
        
        # Send request that will timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                process.send_request("slow_method", {}),
                timeout=2
            )
        
        # Cancel read task
        read_task.cancel()
        try:
            await read_task
        except asyncio.CancelledError:
            pass
    
    @pytest.mark.asyncio
    async def test_process_graceful_shutdown(self, integration_config: BridgeConfig):
        """Test graceful shutdown of MCP process."""
        process = MCPProcess("test-session", integration_config)
        
        mock_process = AsyncMock()
        mock_process.pid = 12345
        mock_process.returncode = None
        mock_process.terminate = AsyncMock()
        mock_process.wait = AsyncMock()
        mock_process.kill = AsyncMock()
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.stderr = AsyncMock()
        
        process.process = mock_process
        process._running = True
        
        # Test graceful shutdown
        await process.stop()
        
        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called_once()
        assert not process._running
        assert process.process is None
    
    @pytest.mark.asyncio
    async def test_process_force_kill(self, integration_config: BridgeConfig):
        """Test force killing MCP process after timeout."""
        process = MCPProcess("test-session", integration_config)
        
        mock_process = AsyncMock()
        mock_process.pid = 12345
        mock_process.returncode = None
        mock_process.terminate = AsyncMock()
        mock_process.wait = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_process.kill = AsyncMock()
        
        process.process = mock_process
        process._running = True
        
        # Test force kill after timeout
        await process.stop()
        
        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()
        assert not process._running


class TestMCPProcessManager:
    """Test MCP process manager functionality."""
    
    @pytest.mark.asyncio
    async def test_process_pool_management(self, integration_config: BridgeConfig):
        """Test process pool creation and management."""
        integration_config.enable_process_pool = True
        integration_config.process_pool_size = 3
        
        manager = MCPProcessManager(integration_config)
        
        with patch.object(MCPProcess, "start", new_callable=AsyncMock, return_value=True):
            await manager.start()
            
            # Verify pool was created
            assert len(manager.process_pool) == 3
            assert all(p._running for p in manager.process_pool)
    
    @pytest.mark.asyncio
    async def test_process_creation_with_pool(self, integration_config: BridgeConfig):
        """Test getting process from pool."""
        integration_config.enable_process_pool = True
        integration_config.process_pool_size = 2
        
        manager = MCPProcessManager(integration_config)
        
        # Create mock pool processes
        mock_process1 = Mock(spec=MCPProcess)
        mock_process1._running = True
        mock_process1.session_id = None
        
        mock_process2 = Mock(spec=MCPProcess)
        mock_process2._running = True
        mock_process2.session_id = None
        
        manager.process_pool = [mock_process1, mock_process2]
        
        # Get process from pool
        process = await manager.get_process("session-1")
        
        assert process == mock_process1
        assert mock_process1.session_id == "session-1"
        assert "session-1" in manager.processes
    
    @pytest.mark.asyncio
    async def test_process_limit_enforcement(self, integration_config: BridgeConfig):
        """Test maximum process limit enforcement."""
        integration_config.max_processes = 2
        manager = MCPProcessManager(integration_config)
        
        # Create max processes
        for i in range(2):
            mock_process = Mock(spec=MCPProcess)
            mock_process._running = True
            manager.processes[f"session-{i}"] = mock_process
        
        # Try to create one more
        with pytest.raises(RuntimeError, match="Maximum number of processes"):
            await manager.create_process("session-3")
    
    @pytest.mark.asyncio
    async def test_process_cleanup(self, integration_config: BridgeConfig):
        """Test cleaning up inactive processes."""
        manager = MCPProcessManager(integration_config)
        
        # Create mix of active and inactive processes
        active_process = Mock(spec=MCPProcess)
        active_process._running = True
        active_process.last_activity = asyncio.get_event_loop().time()
        active_process.stop = AsyncMock()
        
        inactive_process = Mock(spec=MCPProcess)
        inactive_process._running = True
        inactive_process.last_activity = asyncio.get_event_loop().time() - 3600  # 1 hour ago
        inactive_process.stop = AsyncMock()
        
        manager.processes = {
            "active": active_process,
            "inactive": inactive_process,
        }
        
        # Run cleanup
        await manager._cleanup_inactive_processes()
        
        # Verify inactive process was stopped
        inactive_process.stop.assert_called_once()
        assert "inactive" not in manager.processes
        assert "active" in manager.processes
    
    @pytest.mark.asyncio
    async def test_get_stats(self, integration_config: BridgeConfig):
        """Test getting process manager statistics."""
        manager = MCPProcessManager(integration_config)
        
        # Create test processes
        process1 = Mock(spec=MCPProcess)
        process1._running = True
        process1.process = Mock(pid=1234)
        
        process2 = Mock(spec=MCPProcess)
        process2._running = True
        process2.process = Mock(pid=5678)
        
        manager.processes = {
            "session-1": process1,
            "session-2": process2,
        }
        
        stats = manager.get_stats()
        
        assert len(stats) == 2
        assert any(s["pid"] == 1234 for s in stats)
        assert any(s["pid"] == 5678 for s in stats)


class TestProtocolTranslation:
    """Test protocol translation between client formats and MCP."""
    
    def test_parse_jsonrpc_request(self):
        """Test parsing JSON-RPC request."""
        translator = MCPProtocolTranslator()
        
        message = {
            "jsonrpc": "2.0",
            "id": "123",
            "method": "tools/search",
            "params": {"query": "test"}
        }
        
        request = translator.parse_client_message(message)
        
        assert isinstance(request, MCPRequest)
        assert request.id == "123"
        assert request.method == "tools/search"
        assert request.params == {"query": "test"}
    
    def test_parse_openai_format(self):
        """Test parsing OpenAI function call format."""
        translator = MCPProtocolTranslator()
        
        message = {
            "function": {
                "name": "search",
                "arguments": '{"query": "test"}'
            },
            "id": "call_456"
        }
        
        request = translator.parse_client_message(message)
        
        assert isinstance(request, MCPRequest)
        assert request.method == "tools/search"
        assert request.params == {"name": "search", "arguments": {"query": "test"}}
    
    def test_parse_anthropic_format(self):
        """Test parsing Anthropic tool use format."""
        translator = MCPProtocolTranslator()
        
        message = {
            "type": "tool_use",
            "id": "toolu_789",
            "name": "analyze",
            "input": {"data": [1, 2, 3]}
        }
        
        request = translator.parse_client_message(message)
        
        assert isinstance(request, MCPRequest)
        assert request.method == "tools/analyze"
        assert request.params == {"name": "analyze", "arguments": {"data": [1, 2, 3]}}
    
    def test_format_response_jsonrpc(self):
        """Test formatting response as JSON-RPC."""
        translator = MCPProtocolTranslator()
        
        response = MCPResponse(
            id="123",
            result={"status": "success", "data": "test"}
        )
        
        formatted = translator.format_response(response, "json-rpc")
        
        assert formatted["jsonrpc"] == "2.0"
        assert formatted["id"] == "123"
        assert formatted["result"] == {"status": "success", "data": "test"}
    
    def test_format_response_openai(self):
        """Test formatting response as OpenAI format."""
        translator = MCPProtocolTranslator()
        
        response = MCPResponse(
            id="call_456",
            result={"content": [{"type": "text", "text": "Result"}]}
        )
        
        formatted = translator.format_response(response, "openai")
        
        assert formatted["role"] == "function"
        assert formatted["name"] == "call_456"
        assert "Result" in formatted["content"]
    
    def test_format_error_response(self):
        """Test formatting error response."""
        translator = MCPProtocolTranslator()
        
        error_response = translator.create_error_response(
            "req-123",
            MCPErrorCode.METHOD_NOT_FOUND,
            "Unknown method: invalid_method",
            {"method": "invalid_method"}
        )
        
        assert error_response.id == "req-123"
        assert error_response.error["code"] == MCPErrorCode.METHOD_NOT_FOUND
        assert "Unknown method" in error_response.error["message"]
        assert error_response.error["data"]["method"] == "invalid_method"
    
    def test_translate_tool_discovery(self):
        """Test translating tool discovery results."""
        translator = MCPProtocolTranslator()
        
        mcp_tools = {
            "tools": [
                {
                    "name": "search",
                    "description": "Search for information",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"}
                        }
                    }
                }
            ]
        }
        
        # Translate to OpenAI format
        openai_tools = translator.translate_tools(mcp_tools["tools"], "openai")
        
        assert len(openai_tools) == 1
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["function"]["name"] == "search"
        assert openai_tools[0]["function"]["description"] == "Search for information"
        assert "parameters" in openai_tools[0]["function"]
    
    def test_validate_request(self):
        """Test request validation."""
        translator = MCPProtocolTranslator()
        
        # Valid request
        valid_request = MCPRequest(
            id="123",
            method="tools/search",
            params={"query": "test"}
        )
        
        assert translator.validate_request(valid_request) is True
        
        # Invalid request (empty method)
        invalid_request = MCPRequest(
            id="456",
            method="",
            params={}
        )
        
        assert translator.validate_request(invalid_request) is False


class TestSessionManagement:
    """Test session management integration."""
    
    @pytest.mark.asyncio
    async def test_session_creation_flow(self, integration_config: BridgeConfig):
        """Test complete session creation flow."""
        process_manager = MCPProcessManager(integration_config)
        session_manager = BridgeSessionManager(integration_config, process_manager)
        
        # Mock process creation
        mock_process = AsyncMock(spec=MCPProcess)
        mock_process.capabilities = {"tools": ["search"]}
        mock_process.server_info = {"name": "test-server"}
        mock_process.process = Mock(pid=1234)
        
        with patch.object(
            process_manager,
            "create_process",
            return_value=mock_process
        ):
            session = await session_manager.create_session(
                client_id="test-client",
                transport="websocket",
                metadata={"user": "test"}
            )
        
        assert session.client_id == "test-client"
        assert session.transport == "websocket"
        assert session.state == SessionState.READY
        assert session.capabilities == {"tools": ["search"]}
        assert session.metadata["user"] == "test"
        assert session.process_id == 1234
    
    @pytest.mark.asyncio
    async def test_session_request_routing(self, integration_config: BridgeConfig):
        """Test routing requests through sessions."""
        process_manager = MCPProcessManager(integration_config)
        session_manager = BridgeSessionManager(integration_config, process_manager)
        
        # Create session
        session = MCPSession(
            session_id="test-session",
            client_id="test-client",
            state=SessionState.READY
        )
        session_manager.sessions["test-session"] = session
        
        # Mock process
        mock_process = AsyncMock(spec=MCPProcess)
        mock_process.send_request = AsyncMock(
            return_value={"result": "success"}
        )
        
        with patch.object(
            process_manager,
            "get_process",
            return_value=mock_process
        ):
            result = await session_manager.send_request(
                "test-session",
                "tools/search",
                {"query": "test"}
            )
        
        assert result == {"result": "success"}
        mock_process.send_request.assert_called_once_with(
            "tools/search",
            {"query": "test"}
        )
    
    @pytest.mark.asyncio
    async def test_session_state_transitions(self, integration_config: BridgeConfig):
        """Test session state transitions."""
        process_manager = MCPProcessManager(integration_config)
        session_manager = BridgeSessionManager(integration_config, process_manager)
        
        session = MCPSession(
            session_id="state-test",
            state=SessionState.INITIALIZING
        )
        session_manager.sessions["state-test"] = session
        
        # Test state transitions
        await session_manager.update_session_state(
            "state-test",
            SessionState.CONNECTED
        )
        assert session.state == SessionState.CONNECTED
        
        await session_manager.update_session_state(
            "state-test",
            SessionState.READY
        )
        assert session.state == SessionState.READY
        
        await session_manager.update_session_state(
            "state-test",
            SessionState.BUSY
        )
        assert session.state == SessionState.BUSY
        
        await session_manager.update_session_state(
            "state-test",
            SessionState.DISCONNECTED
        )
        assert session.state == SessionState.DISCONNECTED
    
    @pytest.mark.asyncio
    async def test_session_cleanup_on_timeout(self, integration_config: BridgeConfig):
        """Test session cleanup when timeout occurs."""
        integration_config.session_timeout = 1  # 1 second timeout
        
        process_manager = MCPProcessManager(integration_config)
        session_manager = BridgeSessionManager(integration_config, process_manager)
        
        # Create expired session
        session = MCPSession(
            session_id="expired-session",
            state=SessionState.READY
        )
        session.last_activity = asyncio.get_event_loop().time() - 10  # 10 seconds ago
        session_manager.sessions["expired-session"] = session
        
        # Mock process cleanup
        mock_process = AsyncMock(spec=MCPProcess)
        mock_process.stop = AsyncMock()
        
        with patch.object(
            process_manager,
            "get_process",
            return_value=mock_process
        ):
            await session_manager._cleanup_expired_sessions()
        
        assert "expired-session" not in session_manager.sessions
        mock_process.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_concurrent_session_handling(self, integration_config: BridgeConfig):
        """Test handling multiple concurrent sessions."""
        process_manager = MCPProcessManager(integration_config)
        session_manager = BridgeSessionManager(integration_config, process_manager)
        
        # Mock process creation
        async def create_mock_process(session_id):
            mock_process = AsyncMock(spec=MCPProcess)
            mock_process.capabilities = {"tools": [f"tool-{session_id}"]}
            mock_process.process = Mock(pid=1000 + int(session_id.split("-")[-1]))
            return mock_process
        
        with patch.object(
            process_manager,
            "create_process",
            side_effect=create_mock_process
        ):
            # Create multiple sessions concurrently
            tasks = [
                session_manager.create_session(
                    client_id=f"client-{i}",
                    transport="http"
                )
                for i in range(5)
            ]
            
            sessions = await asyncio.gather(*tasks)
        
        assert len(sessions) == 5
        assert all(s.state == SessionState.READY for s in sessions)
        assert len(session_manager.sessions) == 5


class TestErrorRecovery:
    """Test error recovery and resilience."""
    
    @pytest.mark.asyncio
    async def test_process_crash_recovery(self, integration_config: BridgeConfig):
        """Test recovery from process crash."""
        process = MCPProcess("crash-test", integration_config)
        
        # Mock crashed process
        mock_process = AsyncMock()
        mock_process.returncode = 1  # Process crashed
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()
        
        process.process = mock_process
        process._running = True
        
        # Detect crash
        is_alive = await process.is_alive()
        assert not is_alive
        
        # Attempt restart
        with patch("asyncio.create_subprocess_exec") as mock_create:
            new_process = AsyncMock()
            new_process.pid = 9999
            new_process.returncode = None
            new_process.stdin = AsyncMock()
            new_process.stdout = AsyncMock()
            new_process.stdout.readline = AsyncMock(
                return_value=json.dumps({
                    "jsonrpc": "2.0",
                    "id": "init-1",
                    "result": {"capabilities": {}}
                }).encode() + b"\n"
            )
            
            mock_create.return_value = new_process
            
            restarted = await process.restart()
            assert restarted
            assert process._running
    
    @pytest.mark.asyncio
    async def test_malformed_message_handling(self, integration_config: BridgeConfig):
        """Test handling of malformed messages from process."""
        process = MCPProcess("malformed-test", integration_config)
        process._running = True
        process._initialized = True
        
        mock_process = AsyncMock()
        mock_process.stdout = AsyncMock()
        
        # Send malformed JSON
        mock_process.stdout.readline = AsyncMock(
            side_effect=[
                b"not valid json\n",
                b'{"partial": \n',  # Incomplete JSON
                json.dumps({"jsonrpc": "2.0", "id": "123", "result": "ok"}).encode() + b"\n",
                b"",  # EOF
            ]
        )
        
        process.process = mock_process
        
        # Should handle malformed messages gracefully
        await process._read_output()
        
        # Process should still be running
        assert process._running
    
    @pytest.mark.asyncio
    async def test_connection_retry_logic(self, integration_config: BridgeConfig):
        """Test connection retry logic for process startup."""
        process = MCPProcess("retry-test", integration_config)
        
        attempt_count = 0
        
        async def mock_create_subprocess(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            
            if attempt_count < 3:
                # First two attempts fail
                raise OSError("Connection refused")
            
            # Third attempt succeeds
            mock_proc = AsyncMock()
            mock_proc.pid = 7777
            mock_proc.returncode = None
            mock_proc.stdin = AsyncMock()
            mock_proc.stdout = AsyncMock()
            mock_proc.stdout.readline = AsyncMock(
                return_value=json.dumps({
                    "jsonrpc": "2.0",
                    "id": "init-1",
                    "result": {"capabilities": {}}
                }).encode() + b"\n"
            )
            return mock_proc
        
        with patch("asyncio.create_subprocess_exec", side_effect=mock_create_subprocess):
            result = await process.start(max_retries=3)
            
            assert result is True
            assert attempt_count == 3
            assert process._running


class TestPerformanceOptimization:
    """Test performance optimizations and resource management."""
    
    @pytest.mark.asyncio
    async def test_request_batching(self, integration_config: BridgeConfig):
        """Test batching multiple requests for efficiency."""
        process = MCPProcess("batch-test", integration_config)
        process._running = True
        process._initialized = True
        
        mock_process = AsyncMock()
        mock_process.stdin = AsyncMock()
        process.process = mock_process
        
        # Enable request batching
        process.enable_batching = True
        process.batch_size = 3
        process.batch_timeout = 0.1
        
        # Queue multiple requests
        requests = [
            ("method1", {"param": 1}),
            ("method2", {"param": 2}),
            ("method3", {"param": 3}),
        ]
        
        # Mock batch response
        batch_response = {
            "jsonrpc": "2.0",
            "result": [
                {"id": "1", "result": "result1"},
                {"id": "2", "result": "result2"},
                {"id": "3", "result": "result3"},
            ]
        }
        
        with patch.object(process, "_send_batch", return_value=batch_response["result"]):
            tasks = [
                process.send_request(method, params)
                for method, params in requests
            ]
            
            results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert all(r is not None for r in results)
    
    @pytest.mark.asyncio
    async def test_connection_pooling(self, integration_config: BridgeConfig):
        """Test connection pooling for process reuse."""
        integration_config.enable_process_pool = True
        integration_config.process_pool_size = 2
        
        manager = MCPProcessManager(integration_config)
        
        # Create pool
        pool_processes = []
        for i in range(2):
            mock_process = Mock(spec=MCPProcess)
            mock_process._running = True
            mock_process.session_id = None
            mock_process.start = AsyncMock(return_value=True)
            pool_processes.append(mock_process)
        
        with patch.object(MCPProcess, "__new__", side_effect=pool_processes):
            await manager.start()
        
        # Verify pool created
        assert len(manager.process_pool) == 2
        
        # Get processes from pool
        proc1 = await manager.get_process("session-1")
        proc2 = await manager.get_process("session-2")
        
        assert proc1 in pool_processes
        assert proc2 in pool_processes
        assert proc1 != proc2
        
        # Release process back to pool
        await manager.release_process("session-1")
        assert proc1.session_id is None
        
        # Reuse released process
        proc3 = await manager.get_process("session-3")
        assert proc3 == proc1  # Same process reused
    
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, integration_config: BridgeConfig):
        """Test monitoring and limiting memory usage."""
        process = MCPProcess("memory-test", integration_config)
        process._running = True
        
        mock_process = Mock()
        mock_process.pid = 5555
        process.process = mock_process
        
        # Mock memory info
        with patch("psutil.Process") as mock_psutil:
            mock_proc_info = Mock()
            mock_proc_info.memory_info.return_value = Mock(rss=100 * 1024 * 1024)  # 100 MB
            mock_psutil.return_value = mock_proc_info
            
            memory_mb = await process.get_memory_usage()
            assert memory_mb == 100
            
            # Check if exceeds limit
            process.memory_limit_mb = 50
            exceeds = await process.check_memory_limit()
            assert exceeds is True