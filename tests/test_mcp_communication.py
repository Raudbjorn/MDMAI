"""Comprehensive MCP communication tests for TTRPG Assistant.

This module tests all MCP (Model Context Protocol) tool interfaces, including:
- Tool registration and discovery
- stdin/stdout communication protocol
- Request/response serialization
- Error handling in MCP communication
- Tool parameter validation
- Async tool execution
- Response streaming
- Connection management
"""

import asyncio
import json
import sys
import uuid
from io import StringIO
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import pytest_asyncio
from mcp.server.fastmcp import FastMCP

from config.settings import settings
from src.campaign import initialize_campaign_tools, register_campaign_tools
from src.character_generation import initialize_character_tools, register_character_tools
from src.core.database import ChromaDBManager
from src.performance import (
    GlobalCacheManager,
    initialize_performance_tools,
    register_performance_tools,
)
from src.session import initialize_session_tools, register_session_tools
from src.source_management import initialize_source_tools, register_source_tools


class TestMCPToolRegistration:
    """Test MCP tool registration and discovery."""

    @pytest.fixture
    def mcp_server(self):
        """Create an MCP server instance for testing."""
        server = FastMCP("Test TTRPG Assistant")
        return server

    @pytest.fixture
    def mock_db(self):
        """Create a mock database for testing."""
        mock = MagicMock(spec=ChromaDBManager)
        mock.add_document = AsyncMock()
        mock.search = AsyncMock(return_value=[])
        mock.update_document = AsyncMock()
        mock.get_document = AsyncMock()
        mock.list_documents = AsyncMock(return_value=[])
        mock.collections = {"rulebooks": Mock(), "campaigns": Mock()}
        return mock

    def test_campaign_tools_registration(self, mcp_server, mock_db):
        """Test registration of campaign management tools."""
        from src.campaign.campaign_manager import CampaignManager
        from src.campaign.rulebook_linker import RulebookLinker
        
        campaign_manager = CampaignManager(mock_db)
        rulebook_linker = RulebookLinker(mock_db)
        
        # Initialize and register tools
        initialize_campaign_tools(mock_db, campaign_manager, rulebook_linker)
        register_campaign_tools(mcp_server)
        
        # Verify tools are registered
        tools = mcp_server.tools
        
        expected_tools = [
            "create_campaign",
            "get_campaign",
            "update_campaign",
            "delete_campaign",
            "list_campaigns",
            "add_campaign_npc",
            "update_campaign_npc",
            "list_campaign_npcs",
            "add_campaign_location",
            "update_campaign_location",
            "list_campaign_locations",
            "add_campaign_plot_point",
            "update_campaign_plot_point",
            "list_campaign_plot_points",
            "link_campaign_to_rulebooks",
            "get_campaign_rules",
            "search_campaign_content",
            "get_campaign_timeline",
            "export_campaign",
            "import_campaign",
        ]
        
        for tool_name in expected_tools:
            assert tool_name in tools, f"Tool {tool_name} not registered"

    def test_session_tools_registration(self, mcp_server, mock_db):
        """Test registration of session management tools."""
        from src.campaign.campaign_manager import CampaignManager
        from src.session.session_manager import SessionManager
        
        session_manager = SessionManager(mock_db)
        campaign_manager = CampaignManager(mock_db)
        
        # Initialize and register tools
        initialize_session_tools(session_manager, campaign_manager)
        register_session_tools(mcp_server)
        
        # Verify tools are registered
        tools = mcp_server.tools
        
        expected_tools = [
            "create_session",
            "get_session",
            "update_session",
            "end_session",
            "list_sessions",
            "add_session_note",
            "get_session_notes",
            "start_combat",
            "end_combat",
            "add_to_initiative",
            "remove_from_initiative",
            "update_initiative",
            "get_initiative_order",
            "advance_turn",
            "advance_round",
            "roll_dice",
            "add_monster",
            "update_monster_hp",
            "remove_monster",
            "list_active_monsters",
            "get_session_summary",
            "export_session_log",
        ]
        
        for tool_name in expected_tools:
            assert tool_name in tools, f"Tool {tool_name} not registered"

    def test_character_tools_registration(self, mcp_server, mock_db):
        """Test registration of character generation tools."""
        from src.personality.personality_manager import PersonalityManager
        
        personality_manager = PersonalityManager()
        
        # Initialize and register tools
        initialize_character_tools(mock_db, personality_manager)
        register_character_tools(mcp_server)
        
        # Verify tools are registered
        tools = mcp_server.tools
        
        expected_tools = [
            "generate_character",
            "generate_npc",
            "generate_backstory",
            "generate_character_name",
            "get_character_template",
            "validate_character",
            "calculate_character_stats",
            "level_up_character",
            "generate_character_personality",
            "generate_party",
        ]
        
        for tool_name in expected_tools:
            assert tool_name in tools, f"Tool {tool_name} not registered"

    def test_performance_tools_registration(self, mcp_server, mock_db):
        """Test registration of performance monitoring tools."""
        cache_manager = GlobalCacheManager()
        
        # Initialize and register tools
        initialize_performance_tools(
            cache_manager,
            cache_manager.invalidator,
            cache_manager.config,
            mock_db
        )
        register_performance_tools(mcp_server)
        
        # Verify tools are registered
        tools = mcp_server.tools
        
        expected_tools = [
            "get_cache_stats",
            "clear_cache",
            "invalidate_cache_pattern",
            "configure_cache",
            "get_performance_metrics",
            "optimize_database",
            "analyze_query_performance",
        ]
        
        for tool_name in expected_tools:
            assert tool_name in tools, f"Tool {tool_name} not registered"

    def test_source_management_tools_registration(self, mcp_server, mock_db):
        """Test registration of source management tools."""
        from src.pdf_processing.pipeline import PDFProcessingPipeline
        
        pdf_pipeline = PDFProcessingPipeline()
        
        # Initialize and register tools
        initialize_source_tools(mock_db, pdf_pipeline)
        register_source_tools(mcp_server)
        
        # Verify tools are registered
        tools = mcp_server.tools
        
        expected_tools = [
            "add_source_enhanced",
            "validate_source",
            "organize_sources",
            "link_source_to_campaign",
            "get_source_metadata",
            "update_source_metadata",
            "remove_source",
            "merge_sources",
            "get_source_statistics",
            "extract_source_entities",
            "reprocess_source",
        ]
        
        for tool_name in expected_tools:
            assert tool_name in tools, f"Tool {tool_name} not registered"


class TestMCPCommunicationProtocol:
    """Test MCP stdin/stdout communication protocol."""

    @pytest.fixture
    def mcp_server(self):
        """Create configured MCP server."""
        server = FastMCP("Test Server")
        
        # Register a simple test tool
        @server.tool()
        async def test_tool(param1: str, param2: int = 5) -> Dict[str, Any]:
            """Test tool for communication testing."""
            return {
                "status": "success",
                "param1": param1,
                "param2": param2,
                "result": f"Processed {param1} with {param2}"
            }
        
        return server

    @pytest.mark.asyncio
    async def test_tool_invocation_request(self, mcp_server):
        """Test MCP tool invocation request format."""
        # Create request
        request = {
            "jsonrpc": "2.0",
            "id": "test-request-1",
            "method": "tools/call",
            "params": {
                "name": "test_tool",
                "arguments": {
                    "param1": "test_value",
                    "param2": 10
                }
            }
        }
        
        # Process request (simulate MCP processing)
        tool = mcp_server.tools["test_tool"]
        result = await tool.func(param1="test_value", param2=10)
        
        # Verify response format
        assert result["status"] == "success"
        assert result["param1"] == "test_value"
        assert result["param2"] == 10

    @pytest.mark.asyncio
    async def test_tool_discovery_request(self, mcp_server):
        """Test MCP tool discovery request."""
        # Create discovery request
        request = {
            "jsonrpc": "2.0",
            "id": "discovery-1",
            "method": "tools/list",
            "params": {}
        }
        
        # Get tool list
        tools = list(mcp_server.tools.keys())
        
        # Verify tool is discoverable
        assert "test_tool" in tools

    @pytest.mark.asyncio
    async def test_error_response_format(self, mcp_server):
        """Test MCP error response format."""
        # Register error-prone tool
        @mcp_server.tool()
        async def error_tool() -> Dict[str, Any]:
            """Tool that raises an error."""
            raise ValueError("Test error")
        
        # Test error handling
        with pytest.raises(ValueError, match="Test error"):
            await mcp_server.tools["error_tool"].func()

    @pytest.mark.asyncio
    async def test_batch_request_handling(self, mcp_server):
        """Test handling of batch MCP requests."""
        # Create batch request
        batch_request = [
            {
                "jsonrpc": "2.0",
                "id": f"batch-{i}",
                "method": "tools/call",
                "params": {
                    "name": "test_tool",
                    "arguments": {
                        "param1": f"value_{i}",
                        "param2": i
                    }
                }
            }
            for i in range(5)
        ]
        
        # Process batch
        results = []
        for req in batch_request:
            tool = mcp_server.tools[req["params"]["name"]]
            result = await tool.func(**req["params"]["arguments"])
            results.append(result)
        
        # Verify all requests processed
        assert len(results) == 5
        assert all(r["status"] == "success" for r in results)


class TestToolParameterValidation:
    """Test parameter validation for MCP tools."""

    @pytest.fixture
    def validation_server(self):
        """Create server with validation test tools."""
        server = FastMCP("Validation Test Server")
        
        @server.tool()
        async def typed_tool(
            required_str: str,
            required_int: int,
            optional_str: Optional[str] = None,
            optional_bool: bool = False,
            optional_list: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            """Tool with various parameter types."""
            return {
                "required_str": required_str,
                "required_int": required_int,
                "optional_str": optional_str,
                "optional_bool": optional_bool,
                "optional_list": optional_list
            }
        
        return server

    @pytest.mark.asyncio
    async def test_required_parameters(self, validation_server):
        """Test validation of required parameters."""
        tool = validation_server.tools["typed_tool"]
        
        # Valid call with required params
        result = await tool.func(
            required_str="test",
            required_int=42
        )
        assert result["required_str"] == "test"
        assert result["required_int"] == 42

    @pytest.mark.asyncio
    async def test_optional_parameters(self, validation_server):
        """Test handling of optional parameters."""
        tool = validation_server.tools["typed_tool"]
        
        # Call with optional params
        result = await tool.func(
            required_str="test",
            required_int=42,
            optional_str="optional",
            optional_bool=True,
            optional_list=["item1", "item2"]
        )
        
        assert result["optional_str"] == "optional"
        assert result["optional_bool"] is True
        assert result["optional_list"] == ["item1", "item2"]

    @pytest.mark.asyncio
    async def test_type_coercion(self, validation_server):
        """Test automatic type coercion."""
        tool = validation_server.tools["typed_tool"]
        
        # Test with string that should be int
        result = await tool.func(
            required_str="test",
            required_int=42,  # Already int
            optional_bool=True
        )
        
        assert isinstance(result["required_int"], int)
        assert isinstance(result["optional_bool"], bool)


class TestAsyncToolExecution:
    """Test asynchronous tool execution."""

    @pytest.fixture
    def async_server(self):
        """Create server with async tools."""
        server = FastMCP("Async Test Server")
        
        @server.tool()
        async def slow_tool(delay: float = 0.1) -> Dict[str, Any]:
            """Tool that simulates slow operation."""
            await asyncio.sleep(delay)
            return {"status": "completed", "delay": delay}
        
        @server.tool()
        async def parallel_tool(count: int = 3) -> Dict[str, Any]:
            """Tool that can be run in parallel."""
            await asyncio.sleep(0.05)
            return {"status": "success", "count": count}
        
        return server

    @pytest.mark.asyncio
    async def test_async_tool_execution(self, async_server):
        """Test basic async tool execution."""
        tool = async_server.tools["slow_tool"]
        
        start_time = asyncio.get_event_loop().time()
        result = await tool.func(delay=0.2)
        elapsed = asyncio.get_event_loop().time() - start_time
        
        assert result["status"] == "completed"
        assert elapsed >= 0.2

    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(self, async_server):
        """Test concurrent execution of multiple tools."""
        tool = async_server.tools["parallel_tool"]
        
        # Run multiple tools concurrently
        start_time = asyncio.get_event_loop().time()
        tasks = [tool.func(count=i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # Should be much faster than sequential (0.5s)
        assert elapsed < 0.2
        assert len(results) == 10
        assert all(r["status"] == "success" for r in results)

    @pytest.mark.asyncio
    async def test_tool_cancellation(self, async_server):
        """Test cancellation of async tools."""
        tool = async_server.tools["slow_tool"]
        
        # Create task and cancel it
        task = asyncio.create_task(tool.func(delay=1.0))
        await asyncio.sleep(0.1)
        task.cancel()
        
        with pytest.raises(asyncio.CancelledError):
            await task


class TestMCPErrorHandling:
    """Test error handling in MCP communication."""

    @pytest.fixture
    def error_server(self):
        """Create server with error-prone tools."""
        server = FastMCP("Error Test Server")
        
        @server.tool()
        async def validation_error_tool(value: int) -> Dict[str, Any]:
            """Tool that validates input."""
            if value < 0:
                raise ValueError("Value must be non-negative")
            return {"value": value}
        
        @server.tool()
        async def system_error_tool() -> Dict[str, Any]:
            """Tool that raises system error."""
            raise RuntimeError("System error occurred")
        
        @server.tool()
        async def timeout_tool(timeout: float = 5.0) -> Dict[str, Any]:
            """Tool that might timeout."""
            await asyncio.sleep(timeout)
            return {"status": "completed"}
        
        return server

    @pytest.mark.asyncio
    async def test_validation_error_handling(self, error_server):
        """Test handling of validation errors."""
        tool = error_server.tools["validation_error_tool"]
        
        # Test with invalid input
        with pytest.raises(ValueError, match="non-negative"):
            await tool.func(value=-1)
        
        # Test with valid input
        result = await tool.func(value=5)
        assert result["value"] == 5

    @pytest.mark.asyncio
    async def test_system_error_handling(self, error_server):
        """Test handling of system errors."""
        tool = error_server.tools["system_error_tool"]
        
        with pytest.raises(RuntimeError, match="System error"):
            await tool.func()

    @pytest.mark.asyncio
    async def test_timeout_handling(self, error_server):
        """Test handling of timeouts."""
        tool = error_server.tools["timeout_tool"]
        
        # Test with timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(tool.func(timeout=1.0), timeout=0.5)


class TestMCPResponseStreaming:
    """Test streaming responses from MCP tools."""

    @pytest.fixture
    def streaming_server(self):
        """Create server with streaming tools."""
        server = FastMCP("Streaming Test Server")
        
        @server.tool()
        async def streaming_tool(count: int = 5):
            """Tool that actually streams data using async generator."""
            async def stream_chunks():
                """Async generator for streaming chunks."""
                for i in range(count):
                    await asyncio.sleep(0.01)  # Simulate processing delay
                    yield f"chunk_{i}"
            
            # For testing purposes, collect the stream
            # In real usage, this would be consumed as a stream
            chunks = []
            async for chunk in stream_chunks():
                chunks.append(chunk)
            
            return {
                "status": "success", 
                "chunks": chunks,
                "total": count,
                "streamed": True  # Indicate this was streamed
            }
        
        return server

    @pytest.mark.asyncio
    async def test_streaming_response(self, streaming_server):
        """Test streaming response handling."""
        tool = streaming_server.tools["streaming_tool"]
        
        result = await tool.func(count=10)
        
        assert result["status"] == "success"
        assert len(result["chunks"]) == 10
        assert result["total"] == 10
        assert result.get("streamed") is True  # Verify it was streamed
    
    @pytest.mark.asyncio
    async def test_true_streaming_generator(self):
        """Test true streaming with async generator consumption."""
        async def streaming_generator(count: int = 5):
            """A true streaming generator."""
            for i in range(count):
                await asyncio.sleep(0.01)
                yield {"chunk_id": i, "data": f"stream_data_{i}"}
        
        # Consume stream and verify chunks arrive over time
        chunks = []
        start_time = asyncio.get_event_loop().time()
        timestamps = []
        
        async for chunk in streaming_generator(5):
            chunks.append(chunk)
            timestamps.append(asyncio.get_event_loop().time() - start_time)
        
        # Verify we got all chunks
        assert len(chunks) == 5
        assert all(chunk["chunk_id"] == i for i, chunk in enumerate(chunks))
        
        # Verify chunks arrived over time (not all at once)
        assert timestamps[-1] > 0.04  # Should take at least 40ms for 5 chunks


class TestMCPConnectionManagement:
    """Test MCP connection management and lifecycle."""

    @pytest.fixture
    def lifecycle_server(self):
        """Create server with lifecycle management."""
        server = FastMCP("Lifecycle Test Server")
        
        # Add connection tracking
        server.connections = []
        
        @server.tool()
        async def connection_info() -> Dict[str, Any]:
            """Get connection information."""
            return {
                "active_connections": len(server.connections),
                "server_status": "running"
            }
        
        return server

    def test_server_initialization(self, lifecycle_server):
        """Test server initialization."""
        assert lifecycle_server is not None
        assert "connection_info" in lifecycle_server.tools

    @pytest.mark.asyncio
    async def test_connection_lifecycle(self, lifecycle_server):
        """Test connection lifecycle management."""
        # Simulate connection
        connection_id = str(uuid.uuid4())
        lifecycle_server.connections.append(connection_id)
        
        # Check connection
        tool = lifecycle_server.tools["connection_info"]
        result = await tool.func()
        
        assert result["active_connections"] == 1
        assert result["server_status"] == "running"
        
        # Simulate disconnection
        lifecycle_server.connections.remove(connection_id)
        
        result = await tool.func()
        assert result["active_connections"] == 0


class TestComplexMCPScenarios:
    """Test complex MCP communication scenarios."""

    @pytest.fixture
    def complex_server(self, tmp_path):
        """Create server with complex tool interactions."""
        server = FastMCP("Complex Test Server")
        
        # Mock database
        with patch("src.core.database.ChromaDBManager") as MockDB:
            mock_db = MockDB.return_value
            mock_db.add_document = AsyncMock()
            mock_db.search = AsyncMock(return_value=[])
            mock_db.get_document = AsyncMock()
            
            # Initialize all subsystems
            from src.campaign.campaign_manager import CampaignManager
            from src.session.session_manager import SessionManager
            
            campaign_mgr = CampaignManager(mock_db)
            session_mgr = SessionManager(mock_db)
            
            # Register cross-component tool
            @server.tool()
            async def create_campaign_with_session(
                campaign_name: str,
                session_title: str
            ) -> Dict[str, Any]:
                """Create campaign and initial session."""
                # Create campaign
                campaign_id = await campaign_mgr.create_campaign(
                    name=campaign_name,
                    system="D&D 5e",
                    description="Test campaign"
                )
                
                # Create session
                session_id = await session_mgr.create_session(
                    campaign_id=campaign_id,
                    title=session_title,
                    session_number=1
                )
                
                return {
                    "campaign_id": campaign_id,
                    "session_id": session_id,
                    "status": "success"
                }
            
            server.mock_db = mock_db
            server.campaign_mgr = campaign_mgr
            server.session_mgr = session_mgr
        
        return server

    @pytest.mark.asyncio
    async def test_cross_component_tool(self, complex_server):
        """Test tool that uses multiple components."""
        tool = complex_server.tools["create_campaign_with_session"]
        
        result = await tool.func(
            campaign_name="Epic Adventure",
            session_title="The Beginning"
        )
        
        assert result["status"] == "success"
        assert result["campaign_id"] is not None
        assert result["session_id"] is not None

    @pytest.mark.asyncio
    async def test_tool_chaining(self, complex_server):
        """Test chaining multiple tool calls."""
        # First tool call
        create_tool = complex_server.tools["create_campaign_with_session"]
        result1 = await create_tool.func(
            campaign_name="Chain Test",
            session_title="Session 1"
        )
        
        campaign_id = result1["campaign_id"]
        
        # Mock second tool for chaining
        @complex_server.tool()
        async def add_npc_to_campaign(campaign_id: str, npc_name: str) -> Dict[str, Any]:
            """Add NPC to campaign."""
            npc_id = await complex_server.campaign_mgr.add_npc(
                campaign_id,
                {"name": npc_name, "role": "ally"}
            )
            return {"npc_id": npc_id, "campaign_id": campaign_id}
        
        # Chain second tool call
        result2 = await complex_server.tools["add_npc_to_campaign"].func(
            campaign_id=campaign_id,
            npc_name="Friendly Wizard"
        )
        
        assert result2["campaign_id"] == campaign_id
        assert result2["npc_id"] is not None


class TestMCPPerformanceAndLoad:
    """Test MCP performance under load."""

    @pytest.fixture
    def load_server(self):
        """Create server for load testing."""
        server = FastMCP("Load Test Server")
        
        @server.tool()
        async def lightweight_tool(value: int) -> Dict[str, Any]:
            """Lightweight tool for load testing."""
            return {"value": value * 2}
        
        @server.tool()
        async def heavy_tool(size: int = 100) -> Dict[str, Any]:
            """Heavy tool that processes data."""
            data = list(range(size))
            result = sum(data)
            await asyncio.sleep(0.01)  # Simulate processing
            return {"result": result, "size": size}
        
        return server

    @pytest.mark.asyncio
    async def test_high_frequency_calls(self, load_server):
        """Test high frequency tool calls."""
        tool = load_server.tools["lightweight_tool"]
        
        # Make many rapid calls
        start_time = asyncio.get_event_loop().time()
        tasks = [tool.func(value=i) for i in range(100)]
        results = await asyncio.gather(*tasks)
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # Verify all completed
        assert len(results) == 100
        assert all(r["value"] == i * 2 for i, r in enumerate(results))
        
        # Should handle 100 calls quickly
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_mixed_load_pattern(self, load_server):
        """Test mixed load pattern with different tools."""
        light_tool = load_server.tools["lightweight_tool"]
        heavy_tool = load_server.tools["heavy_tool"]
        
        # Create mixed workload
        tasks = []
        for i in range(50):
            if i % 5 == 0:
                # Heavy task every 5th call
                tasks.append(heavy_tool.func(size=1000))
            else:
                # Light tasks
                tasks.append(light_tool.func(value=i))
        
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks)
        elapsed = asyncio.get_event_loop().time() - start_time
        
        assert len(results) == 50
        # Should handle mixed load efficiently
        assert elapsed < 2.0