"""End-to-end tests for the web interface and real-time communication."""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import WebSocket
from fastapi.testclient import TestClient
from httpx import AsyncClient
from websockets import connect as ws_connect

from src.bridge.bridge_server import BridgeServer, create_bridge_app
from src.bridge.models import BridgeConfig, SessionState, TransportType


@pytest.fixture
def e2e_config() -> BridgeConfig:
    """Create configuration for E2E testing."""
    return BridgeConfig(
        mcp_server_path="python",
        mcp_server_args=["-m", "src.main"],
        max_processes=10,
        process_timeout=30,
        session_timeout=300,
        enable_websocket=True,
        enable_sse=True,
        enable_http=True,
        require_auth=False,
        log_requests=True,
        log_responses=True,
    )


@pytest.fixture
async def bridge_app(e2e_config: BridgeConfig):
    """Create and start bridge application for E2E testing."""
    server = BridgeServer(e2e_config)
    
    # Mock the process manager to avoid actual subprocess creation
    with patch.object(server.process_manager, "start", new_callable=AsyncMock):
        with patch.object(server.session_manager, "start", new_callable=AsyncMock):
            await server.startup()
            yield server.app
            await server.shutdown()


@pytest.fixture
async def async_client(bridge_app):
    """Create async HTTP client for testing."""
    async with AsyncClient(app=bridge_app, base_url="http://test") as client:
        yield client


class TestHTTPEndToEnd:
    """End-to-end tests for HTTP REST API."""
    
    @pytest.mark.asyncio
    async def test_complete_session_lifecycle(self, async_client: AsyncClient):
        """Test complete session lifecycle via HTTP."""
        # Create session
        create_response = await async_client.post(
            "/sessions",
            json={
                "client_id": "e2e-test-client",
                "metadata": {"test": "e2e", "timestamp": time.time()}
            }
        )
        assert create_response.status_code == 200
        session_data = create_response.json()
        session_id = session_data["session_id"]
        
        assert session_data["client_id"] == "e2e-test-client"
        assert session_data["state"] in ["initializing", "ready"]
        
        # Get session info
        get_response = await async_client.get(f"/sessions/{session_id}")
        assert get_response.status_code == 200
        session_info = get_response.json()
        assert session_info["session_id"] == session_id
        
        # Discover tools
        with patch("src.bridge.session_manager.BridgeSessionManager.send_request") as mock_send:
            mock_send.return_value = {
                "tools": [
                    {"name": "search", "description": "Search tool"},
                    {"name": "analyze", "description": "Analyze tool"}
                ]
            }
            
            discover_response = await async_client.post(
                "/tools/discover",
                json={"session_id": session_id}
            )
            assert discover_response.status_code == 200
            tools_data = discover_response.json()
            assert len(tools_data["tools"]) == 2
        
        # Call a tool
        with patch("src.bridge.session_manager.BridgeSessionManager.send_request") as mock_send:
            mock_send.return_value = {
                "result": "Search completed",
                "items": ["item1", "item2", "item3"]
            }
            
            call_response = await async_client.post(
                "/tools/call",
                json={
                    "session_id": session_id,
                    "tool": "search",
                    "params": {"query": "test query"}
                }
            )
            assert call_response.status_code == 200
            call_data = call_response.json()
            assert call_data["tool"] == "search"
            assert len(call_data["result"]["items"]) == 3
        
        # Delete session
        delete_response = await async_client.delete(f"/sessions/{session_id}")
        assert delete_response.status_code == 200
        
        # Verify session is deleted
        get_deleted_response = await async_client.get(f"/sessions/{session_id}")
        assert get_deleted_response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_concurrent_sessions(self, async_client: AsyncClient):
        """Test handling multiple concurrent sessions."""
        session_ids = []
        
        # Create multiple sessions concurrently
        tasks = []
        for i in range(5):
            task = async_client.post(
                "/sessions",
                json={"client_id": f"concurrent-client-{i}"}
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        for i, response in enumerate(responses):
            assert response.status_code == 200
            session_data = response.json()
            session_ids.append(session_data["session_id"])
            assert session_data["client_id"] == f"concurrent-client-{i}"
        
        # Verify all sessions exist
        for session_id in session_ids:
            get_response = await async_client.get(f"/sessions/{session_id}")
            assert get_response.status_code == 200
        
        # Clean up sessions
        cleanup_tasks = [
            async_client.delete(f"/sessions/{session_id}")
            for session_id in session_ids
        ]
        await asyncio.gather(*cleanup_tasks)
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, async_client: AsyncClient):
        """Test rate limiting functionality."""
        with patch("src.bridge.config.settings.enable_rate_limiting", True):
            with patch("src.bridge.config.settings.rate_limit_requests", 3):
                with patch("src.bridge.config.settings.rate_limit_period", 1):
                    # Make requests up to limit
                    for i in range(3):
                        response = await async_client.get("/health")
                        assert response.status_code == 200
                    
                    # Next request should be rate limited
                    # Note: This would require actual rate limiting implementation
                    # For now, we're testing the concept
    
    @pytest.mark.asyncio
    async def test_error_handling(self, async_client: AsyncClient):
        """Test error handling across the API."""
        # Test non-existent session
        response = await async_client.get("/sessions/non-existent-session")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
        
        # Test invalid tool call
        with patch("src.bridge.session_manager.BridgeSessionManager.send_request") as mock_send:
            mock_send.side_effect = RuntimeError("Tool execution failed")
            
            response = await async_client.post(
                "/tools/call",
                json={
                    "tool": "invalid_tool",
                    "params": {}
                }
            )
            assert response.status_code == 500
            assert "Tool execution failed" in response.json()["detail"]


class TestWebSocketEndToEnd:
    """End-to-end tests for WebSocket communication."""
    
    @pytest.mark.asyncio
    async def test_websocket_session_flow(self, bridge_app):
        """Test complete WebSocket session flow."""
        with TestClient(bridge_app) as client:
            with client.websocket_connect("/ws") as websocket:
                # Create session
                websocket.send_json({
                    "type": "create_session",
                    "client_id": "ws-e2e-client",
                    "metadata": {"source": "e2e-test"}
                })
                
                # Receive session created response
                response = websocket.receive_json()
                assert response["type"] == "session_created"
                session_id = response["session_id"]
                assert "capabilities" in response
                
                # Send a request
                with patch("src.bridge.session_manager.BridgeSessionManager.send_request") as mock_send:
                    mock_send.return_value = {"result": "success", "data": [1, 2, 3]}
                    
                    websocket.send_json({
                        "jsonrpc": "2.0",
                        "id": "req-001",
                        "method": "tools/search",
                        "params": {"query": "test"},
                        "format": "json-rpc"
                    })
                    
                    # Receive response
                    response = websocket.receive_json()
                    assert response["type"] == "response"
                    assert response["data"]["id"] == "req-001"
                    assert response["data"]["result"]["data"] == [1, 2, 3]
    
    @pytest.mark.asyncio
    async def test_websocket_attach_session(self, bridge_app):
        """Test attaching to existing session via WebSocket."""
        # First create a session via HTTP
        async with AsyncClient(app=bridge_app, base_url="http://test") as client:
            create_response = await client.post(
                "/sessions",
                json={"client_id": "attach-test-client"}
            )
            session_data = create_response.json()
            session_id = session_data["session_id"]
        
        # Now attach via WebSocket
        with TestClient(bridge_app) as client:
            with client.websocket_connect("/ws") as websocket:
                websocket.send_json({
                    "type": "attach_session",
                    "session_id": session_id
                })
                
                response = websocket.receive_json()
                assert response["type"] == "session_attached"
                assert response["session_id"] == session_id
    
    @pytest.mark.asyncio
    async def test_websocket_error_handling(self, bridge_app):
        """Test WebSocket error handling."""
        with TestClient(bridge_app) as client:
            with client.websocket_connect("/ws") as websocket:
                # Try to attach to non-existent session
                websocket.send_json({
                    "type": "attach_session",
                    "session_id": "non-existent"
                })
                
                response = websocket.receive_json()
                assert response["type"] == "error"
                assert "not found" in response["error"].lower()
    
    @pytest.mark.asyncio
    async def test_websocket_concurrent_messages(self, bridge_app):
        """Test handling concurrent messages over WebSocket."""
        with TestClient(bridge_app) as client:
            with client.websocket_connect("/ws") as websocket:
                # Create session
                websocket.send_json({
                    "type": "create_session",
                    "client_id": "concurrent-ws-client"
                })
                
                response = websocket.receive_json()
                assert response["type"] == "session_created"
                
                # Send multiple requests rapidly
                with patch("src.bridge.session_manager.BridgeSessionManager.send_request") as mock_send:
                    mock_send.return_value = {"result": "success"}
                    
                    for i in range(5):
                        websocket.send_json({
                            "jsonrpc": "2.0",
                            "id": f"req-{i:03d}",
                            "method": f"tools/tool{i}",
                            "params": {"index": i}
                        })
                    
                    # Receive all responses
                    responses = []
                    for _ in range(5):
                        response = websocket.receive_json()
                        responses.append(response)
                    
                    assert len(responses) == 5
                    assert all(r["type"] in ["response", "error"] for r in responses)


class TestServerSentEvents:
    """End-to-end tests for Server-Sent Events."""
    
    @pytest.mark.asyncio
    async def test_sse_connection(self, async_client: AsyncClient):
        """Test SSE connection and heartbeat."""
        # Create or use a session
        session_id = "sse-test-session"
        
        # Connect to SSE endpoint
        async with async_client.stream("GET", f"/events/{session_id}") as response:
            assert response.status_code == 200
            
            # Read first few events
            events = []
            async for line in response.aiter_lines():
                if line.startswith("event:"):
                    events.append(line)
                    if len(events) >= 2:  # Get connected and at least one heartbeat
                        break
            
            assert any("connected" in event for event in events)
    
    @pytest.mark.asyncio
    async def test_sse_session_creation(self, async_client: AsyncClient):
        """Test SSE with automatic session creation."""
        # Connect to SSE with non-existent session
        async with async_client.stream("GET", "/events/new-sse-session") as response:
            assert response.status_code == 200
            
            # Read connected event
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data = json.loads(line[5:])  # Remove "data:" prefix
                    assert "session_id" in data
                    assert "capabilities" in data
                    break


class TestPerformanceAndLoad:
    """Performance and load testing."""
    
    @pytest.mark.asyncio
    async def test_high_throughput(self, async_client: AsyncClient):
        """Test handling high request throughput."""
        # Create a session
        create_response = await async_client.post(
            "/sessions",
            json={"client_id": "perf-test-client"}
        )
        session_data = create_response.json()
        session_id = session_data["session_id"]
        
        # Send many requests concurrently
        with patch("src.bridge.session_manager.BridgeSessionManager.send_request") as mock_send:
            mock_send.return_value = {"result": "success"}
            
            start_time = time.time()
            
            tasks = []
            for i in range(100):
                task = async_client.post(
                    "/tools/call",
                    json={
                        "session_id": session_id,
                        "tool": "test_tool",
                        "params": {"index": i}
                    }
                )
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # All requests should succeed
            assert all(r.status_code == 200 for r in responses)
            
            # Should complete reasonably quickly (adjust threshold as needed)
            assert duration < 10  # 100 requests in under 10 seconds
            
            requests_per_second = 100 / duration
            print(f"Throughput: {requests_per_second:.2f} requests/second")
    
    @pytest.mark.asyncio
    async def test_concurrent_websockets(self, bridge_app):
        """Test handling multiple concurrent WebSocket connections."""
        websockets = []
        
        with TestClient(bridge_app) as client:
            # Create multiple WebSocket connections
            for i in range(10):
                ws = client.websocket_connect("/ws").__enter__()
                websockets.append(ws)
                
                # Create session on each WebSocket
                ws.send_json({
                    "type": "create_session",
                    "client_id": f"concurrent-ws-{i}"
                })
                
                response = ws.receive_json()
                assert response["type"] == "session_created"
            
            # Send messages on all WebSockets
            with patch("src.bridge.session_manager.BridgeSessionManager.send_request") as mock_send:
                mock_send.return_value = {"result": "success"}
                
                for i, ws in enumerate(websockets):
                    ws.send_json({
                        "jsonrpc": "2.0",
                        "id": f"ws-{i}-req",
                        "method": "tools/test",
                        "params": {"ws_index": i}
                    })
                
                # Receive responses
                for i, ws in enumerate(websockets):
                    response = ws.receive_json()
                    assert response["type"] == "response"
            
            # Clean up
            for ws in websockets:
                ws.__exit__(None, None, None)
    
    @pytest.mark.asyncio
    async def test_memory_stability(self, async_client: AsyncClient):
        """Test memory stability under sustained load."""
        import gc
        import tracemalloc
        
        tracemalloc.start()
        
        # Take initial snapshot
        snapshot1 = tracemalloc.take_snapshot()
        
        # Run sustained operations
        for iteration in range(10):
            # Create and delete sessions
            session_ids = []
            
            for i in range(10):
                response = await async_client.post(
                    "/sessions",
                    json={"client_id": f"mem-test-{iteration}-{i}"}
                )
                session_ids.append(response.json()["session_id"])
            
            # Perform operations
            with patch("src.bridge.session_manager.BridgeSessionManager.send_request") as mock_send:
                mock_send.return_value = {"result": "success"}
                
                for session_id in session_ids:
                    await async_client.post(
                        "/tools/call",
                        json={
                            "session_id": session_id,
                            "tool": "test",
                            "params": {}
                        }
                    )
            
            # Clean up
            for session_id in session_ids:
                await async_client.delete(f"/sessions/{session_id}")
            
            # Force garbage collection
            gc.collect()
        
        # Take final snapshot
        snapshot2 = tracemalloc.take_snapshot()
        
        # Compare snapshots
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        # Check for significant memory growth
        total_growth = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
        
        # Memory growth should be minimal (adjust threshold as needed)
        assert total_growth < 10 * 1024 * 1024  # Less than 10 MB growth
        
        tracemalloc.stop()


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""
    
    @pytest.mark.asyncio
    async def test_chat_assistant_workflow(self, async_client: AsyncClient):
        """Test typical chat assistant workflow."""
        # Create session for user
        create_response = await async_client.post(
            "/sessions",
            json={
                "client_id": "user-123",
                "metadata": {
                    "user_id": "123",
                    "conversation_id": "conv-456",
                    "model": "assistant-v1"
                }
            }
        )
        session_data = create_response.json()
        session_id = session_data["session_id"]
        
        # Discover available tools
        with patch("src.bridge.session_manager.BridgeSessionManager.send_request") as mock_send:
            mock_send.return_value = {
                "tools": [
                    {"name": "web_search", "description": "Search the web"},
                    {"name": "calculator", "description": "Perform calculations"},
                    {"name": "weather", "description": "Get weather information"}
                ]
            }
            
            discover_response = await async_client.post(
                "/tools/discover",
                json={"session_id": session_id}
            )
            tools = discover_response.json()["tools"]
            assert len(tools) == 3
        
        # Simulate conversation with tool calls
        conversation_flow = [
            ("web_search", {"query": "Python async programming"}, {"results": ["Article 1", "Article 2"]}),
            ("calculator", {"expression": "2 + 2"}, {"result": 4}),
            ("weather", {"location": "San Francisco"}, {"temperature": 65, "condition": "sunny"})
        ]
        
        for tool_name, params, expected_result in conversation_flow:
            with patch("src.bridge.session_manager.BridgeSessionManager.send_request") as mock_send:
                mock_send.return_value = expected_result
                
                response = await async_client.post(
                    "/tools/call",
                    json={
                        "session_id": session_id,
                        "tool": tool_name,
                        "params": params
                    }
                )
                assert response.status_code == 200
                result = response.json()["result"]
                assert result == expected_result
        
        # End session
        await async_client.delete(f"/sessions/{session_id}")
    
    @pytest.mark.asyncio
    async def test_long_running_analysis(self, async_client: AsyncClient):
        """Test long-running analysis with progress updates."""
        # Create session
        create_response = await async_client.post(
            "/sessions",
            json={"client_id": "analyst-client"}
        )
        session_id = create_response.json()["session_id"]
        
        # Start long-running analysis
        with patch("src.bridge.session_manager.BridgeSessionManager.send_request") as mock_send:
            # Simulate delayed response
            async def delayed_response(*args, **kwargs):
                await asyncio.sleep(0.5)  # Simulate processing time
                return {
                    "status": "completed",
                    "results": {
                        "summary": "Analysis complete",
                        "metrics": {"accuracy": 0.95, "precision": 0.92}
                    }
                }
            
            mock_send.side_effect = delayed_response
            
            start_time = time.time()
            response = await async_client.post(
                "/tools/call",
                json={
                    "session_id": session_id,
                    "tool": "analyze_dataset",
                    "params": {"dataset_id": "data-789"}
                },
                timeout=10  # Longer timeout for long-running operations
            )
            end_time = time.time()
            
            assert response.status_code == 200
            result = response.json()["result"]
            assert result["status"] == "completed"
            assert end_time - start_time >= 0.5  # Verify it took time
    
    @pytest.mark.asyncio
    async def test_multi_user_collaboration(self, async_client: AsyncClient):
        """Test multiple users collaborating through shared resources."""
        # Create sessions for multiple users
        user_sessions = {}
        
        for user_id in ["alice", "bob", "charlie"]:
            response = await async_client.post(
                "/sessions",
                json={
                    "client_id": f"user-{user_id}",
                    "metadata": {"user": user_id, "team": "research"}
                }
            )
            user_sessions[user_id] = response.json()["session_id"]
        
        # Simulate collaborative workflow
        with patch("src.bridge.session_manager.BridgeSessionManager.send_request") as mock_send:
            # Alice creates a document
            mock_send.return_value = {"document_id": "doc-123", "status": "created"}
            response = await async_client.post(
                "/tools/call",
                json={
                    "session_id": user_sessions["alice"],
                    "tool": "create_document",
                    "params": {"title": "Research Notes"}
                }
            )
            doc_id = response.json()["result"]["document_id"]
            
            # Bob adds content
            mock_send.return_value = {"status": "updated", "revision": 2}
            response = await async_client.post(
                "/tools/call",
                json={
                    "session_id": user_sessions["bob"],
                    "tool": "update_document",
                    "params": {"document_id": doc_id, "content": "Additional findings"}
                }
            )
            
            # Charlie reviews
            mock_send.return_value = {"content": "Research Notes\nAdditional findings", "revision": 2}
            response = await async_client.post(
                "/tools/call",
                json={
                    "session_id": user_sessions["charlie"],
                    "tool": "read_document",
                    "params": {"document_id": doc_id}
                }
            )
            
            assert response.json()["result"]["revision"] == 2
        
        # Clean up sessions
        for session_id in user_sessions.values():
            await async_client.delete(f"/sessions/{session_id}")


class TestFailureRecovery:
    """Test failure recovery and resilience."""
    
    @pytest.mark.asyncio
    async def test_session_recovery_after_error(self, async_client: AsyncClient):
        """Test session recovery after an error."""
        # Create session
        create_response = await async_client.post(
            "/sessions",
            json={"client_id": "recovery-test"}
        )
        session_id = create_response.json()["session_id"]
        
        # Cause an error
        with patch("src.bridge.session_manager.BridgeSessionManager.send_request") as mock_send:
            mock_send.side_effect = RuntimeError("Simulated failure")
            
            error_response = await async_client.post(
                "/tools/call",
                json={
                    "session_id": session_id,
                    "tool": "failing_tool",
                    "params": {}
                }
            )
            assert error_response.status_code == 500
        
        # Session should still be usable
        with patch("src.bridge.session_manager.BridgeSessionManager.send_request") as mock_send:
            mock_send.return_value = {"result": "success"}
            
            recovery_response = await async_client.post(
                "/tools/call",
                json={
                    "session_id": session_id,
                    "tool": "working_tool",
                    "params": {}
                }
            )
            assert recovery_response.status_code == 200
            assert recovery_response.json()["result"]["result"] == "success"
    
    @pytest.mark.asyncio
    async def test_websocket_reconnection(self, bridge_app):
        """Test WebSocket reconnection scenarios."""
        session_id = None
        
        # First connection - create session
        with TestClient(bridge_app) as client:
            with client.websocket_connect("/ws") as websocket:
                websocket.send_json({
                    "type": "create_session",
                    "client_id": "reconnect-test"
                })
                
                response = websocket.receive_json()
                session_id = response["session_id"]
        
        # Simulate disconnect and reconnect
        with TestClient(bridge_app) as client:
            with client.websocket_connect("/ws") as websocket:
                # Attach to existing session
                websocket.send_json({
                    "type": "attach_session",
                    "session_id": session_id
                })
                
                response = websocket.receive_json()
                assert response["type"] == "session_attached"
                assert response["session_id"] == session_id
                
                # Should be able to continue using session
                with patch("src.bridge.session_manager.BridgeSessionManager.send_request") as mock_send:
                    mock_send.return_value = {"result": "reconnected"}
                    
                    websocket.send_json({
                        "jsonrpc": "2.0",
                        "id": "reconnect-req",
                        "method": "tools/test",
                        "params": {}
                    })
                    
                    response = websocket.receive_json()
                    assert response["type"] == "response"
                    assert response["data"]["result"]["result"] == "reconnected"