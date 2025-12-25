"""Pytest configuration and shared fixtures for all tests."""

import asyncio
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.bridge.bridge_server import BridgeServer, create_bridge_app
from src.bridge.mcp_process_manager import MCPProcess, MCPProcessManager
from src.bridge.models import BridgeConfig, MCPSession, SessionState, TransportType
from src.bridge.protocol_translator import MCPProtocolTranslator
from src.bridge.session_manager import BridgeSessionManager


# Configure pytest-asyncio
pytest_asyncio.fixture = pytest.fixture


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config() -> BridgeConfig:
    """Create a test configuration."""
    return BridgeConfig(
        mcp_server_path="python",
        mcp_server_args=["-m", "src.main"],
        max_processes=5,
        process_timeout=30,
        session_timeout=300,
        max_sessions_per_client=3,
        enable_websocket=True,
        enable_sse=True,
        enable_http=True,
        require_auth=False,
        enable_rate_limiting=False,
        log_requests=False,
        log_responses=False,
    )


@pytest.fixture
def mock_process() -> Mock:
    """Create a mock MCP process."""
    process = Mock(spec=MCPProcess)
    process._running = True
    process._initialized = True
    process.session_id = "test-session"
    process.capabilities = {"tools": ["search", "analyze"]}
    process.server_info = {"name": "test-server", "version": "1.0.0"}
    process.process = Mock(pid=12345)
    process.start = AsyncMock(return_value=True)
    process.stop = AsyncMock()
    process.send_request = AsyncMock(return_value={"result": "success"})
    process.is_alive = AsyncMock(return_value=True)
    process.restart = AsyncMock(return_value=True)
    process.get_memory_usage = AsyncMock(return_value=50.0)
    process.check_memory_limit = AsyncMock(return_value=False)
    return process


@pytest.fixture
def mock_process_manager(test_config: BridgeConfig, mock_process: Mock) -> Mock:
    """Create a mock process manager."""
    manager = Mock(spec=MCPProcessManager)
    manager.config = test_config
    manager.processes = {}
    manager.process_pool = []
    manager.start = AsyncMock()
    manager.stop = AsyncMock()
    manager.create_process = AsyncMock(return_value=mock_process)
    manager.get_process = AsyncMock(return_value=mock_process)
    manager.remove_process = AsyncMock()
    manager.release_process = AsyncMock()
    manager.get_stats = Mock(return_value=[])
    return manager


@pytest.fixture
def mock_session_manager(test_config: BridgeConfig, mock_process_manager: Mock) -> Mock:
    """Create a mock session manager."""
    manager = Mock(spec=BridgeSessionManager)
    manager.config = test_config
    manager.process_manager = mock_process_manager
    manager.sessions = {}
    manager.start = AsyncMock()
    manager.stop = AsyncMock()
    manager.create_session = AsyncMock(
        return_value=MCPSession(
            session_id="test-session-123",
            client_id="test-client",
            state=SessionState.READY,
            transport=TransportType.HTTP,
            capabilities={"tools": ["search"]},
        )
    )
    manager.get_session = AsyncMock(return_value=None)
    manager.remove_session = AsyncMock()
    manager.update_session_state = AsyncMock()
    manager.send_request = AsyncMock(return_value={"result": "success"})
    manager.get_stats = Mock(return_value={"active_sessions": 0})
    return manager


@pytest.fixture
def mock_translator() -> Mock:
    """Create a mock protocol translator."""
    translator = Mock(spec=MCPProtocolTranslator)
    translator.parse_client_message = Mock()
    translator.format_response = Mock()
    translator.create_error_response = Mock()
    translator.translate_tools = Mock()
    translator.validate_request = Mock(return_value=True)
    return translator


@pytest.fixture
def bridge_server(test_config: BridgeConfig) -> BridgeServer:
    """Create a bridge server for testing."""
    server = BridgeServer(test_config)
    
    # Mock the managers to avoid subprocess creation
    server.process_manager = Mock(spec=MCPProcessManager)
    server.process_manager.start = AsyncMock()
    server.process_manager.stop = AsyncMock()
    server.process_manager.get_stats = Mock(return_value=[])
    
    server.session_manager = Mock(spec=BridgeSessionManager)
    server.session_manager.start = AsyncMock()
    server.session_manager.stop = AsyncMock()
    server.session_manager.get_stats = Mock(return_value={"active_sessions": 0})
    
    return server


@pytest.fixture
def app(bridge_server: BridgeServer):
    """Create FastAPI app for testing."""
    return bridge_server.app


@pytest.fixture
def client(app) -> TestClient:
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
async def async_client(app) -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_websocket() -> Mock:
    """Create a mock WebSocket connection."""
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.send_text = AsyncMock()
    ws.receive_json = AsyncMock()
    ws.receive_text = AsyncMock()
    ws.close = AsyncMock()
    return ws


# Markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests for complete workflows"
    )
    config.addinivalue_line(
        "markers", "load: Load and performance tests"
    )
    config.addinivalue_line(
        "markers", "stress: Stress tests for system limits"
    )
    config.addinivalue_line(
        "markers", "security: Security tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )
    config.addinivalue_line(
        "markers", "requires_redis: Tests that require Redis"
    )
    config.addinivalue_line(
        "markers", "requires_docker: Tests that require Docker"
    )


# Test environment setup
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "WARNING"
    os.environ["MCP_SERVER_PATH"] = "python -m src.main"
    yield
    # Cleanup
    os.environ.pop("TESTING", None)


# Cleanup fixtures
@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """Clean up after each test."""
    yield
    # Force garbage collection
    import gc
    gc.collect()

    # Close any remaining async tasks (but not the current task)
    current_task = asyncio.current_task()
    for task in asyncio.all_tasks():
        if task is not current_task and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=0.1)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass


# Utility fixtures
@pytest.fixture
def sample_mcp_request() -> dict:
    """Create a sample MCP request."""
    return {
        "jsonrpc": "2.0",
        "id": "test-123",
        "method": "tools/search",
        "params": {"query": "test query"},
    }


@pytest.fixture
def sample_mcp_response() -> dict:
    """Create a sample MCP response."""
    return {
        "jsonrpc": "2.0",
        "id": "test-123",
        "result": {
            "content": [
                {"type": "text", "text": "Search results"}
            ]
        },
    }


@pytest.fixture
def sample_mcp_error() -> dict:
    """Create a sample MCP error response."""
    return {
        "jsonrpc": "2.0",
        "id": "test-123",
        "error": {
            "code": -32601,
            "message": "Method not found",
            "data": {"method": "unknown"},
        },
    }


@pytest.fixture
def sample_tools() -> list:
    """Create sample MCP tools."""
    return [
        {
            "name": "search",
            "description": "Search for information",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "analyze",
            "description": "Analyze data",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "data": {"type": "array", "description": "Data to analyze"},
                },
            },
        },
    ]


# Performance testing utilities
@pytest.fixture
def performance_timer():
    """Create a performance timer context manager."""
    import time
    from contextlib import contextmanager
    
    @contextmanager
    def timer():
        start = time.perf_counter()
        metrics = {"start": start}
        yield metrics
        end = time.perf_counter()
        metrics["end"] = end
        metrics["duration"] = end - start
    
    return timer


# Mock external services
@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock()
    redis.delete = AsyncMock()
    redis.incr = AsyncMock(return_value=1)
    redis.expire = AsyncMock()
    redis.ping = AsyncMock(return_value=True)
    return redis


@pytest.fixture
def mock_database():
    """Create a mock database connection."""
    db = AsyncMock()
    db.execute = AsyncMock()
    db.fetch_one = AsyncMock(return_value=None)
    db.fetch_all = AsyncMock(return_value=[])
    db.transaction = AsyncMock()
    return db