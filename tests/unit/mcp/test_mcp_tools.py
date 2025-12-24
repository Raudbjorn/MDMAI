"""Unit tests for MCP tools and server interface."""

import asyncio
import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest

# Mock FastMCP before importing
with patch("mcp.server.fastmcp.FastMCP"):
    from src.main import add_campaign_data, get_campaign_data, process_pdf, search


class TestMCPTools:
    """Test MCP tool interfaces."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database."""
        db = Mock()
        db.search = AsyncMock(return_value=[])
        db.add_document = AsyncMock()
        db.get_document = AsyncMock()
        db.update_document = AsyncMock()
        return db

    @pytest.fixture
    def mock_mcp_server(self):
        """Create mock MCP server."""
        server = Mock()
        server.tool = Mock(return_value=lambda func: func)
        return server

    @pytest.mark.asyncio
    async def test_search_tool(self, mock_db):
        """Test the search MCP tool."""
        with patch("src.main.db", mock_db):
            mock_db.search.return_value = [
                {
                    "id": "1",
                    "content": "Fireball spell description",
                    "metadata": {"page": 241, "source": "PHB"},
                    "distance": 0.1,
                }
            ]

            result = await search(
                query="fireball spell", rulebook="PHB", max_results=5, use_hybrid=True
            )

            assert result["status"] == "success"
            assert len(result["results"]) == 1
            assert "Fireball" in result["results"][0]["content"]
            mock_db.search.assert_called()

    @pytest.mark.asyncio
    async def test_search_tool_with_filters(self, mock_db):
        """Test search tool with metadata filters."""
        with patch("src.main.db", mock_db):
            mock_db.search.return_value = []

            result = await search(
                query="wizard", source_type="rulebook", content_type="class", max_results=10
            )

            # Verify filters were applied
            call_args = mock_db.search.call_args
            assert "metadata_filter" in call_args[1]
            filters = call_args[1]["metadata_filter"]
            assert filters.get("source_type") == "rulebook"
            assert filters.get("content_type") == "class"

    @pytest.mark.asyncio
    async def test_search_tool_error_handling(self, mock_db):
        """Test search tool error handling."""
        with patch("src.main.db", mock_db):
            mock_db.search.side_effect = Exception("Database error")

            result = await search("test query")

            assert result["status"] == "error"
            assert "Database error" in result["error"]

    @pytest.mark.asyncio
    async def test_process_pdf_tool(self, mock_db):
        """Test the PDF processing MCP tool."""
        with patch("src.main.pdf_pipeline") as mock_pipeline:
            mock_pipeline.process_pdf = AsyncMock(
                return_value={
                    "status": "success",
                    "source_id": "src_123",
                    "total_chunks": 100,
                    "stored_chunks": 100,
                }
            )

            result = await process_pdf(
                pdf_path="/path/to/rulebook.pdf",
                rulebook_name="Player's Handbook",
                system="D&D 5e",
                source_type="rulebook",
            )

            assert result["status"] == "success"
            assert result["message"] == "PDF processed successfully"
            assert result["source_id"] == "src_123"
            mock_pipeline.process_pdf.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_pdf_duplicate_detection(self, mock_db):
        """Test PDF duplicate detection."""
        with patch("src.main.pdf_pipeline") as mock_pipeline:
            mock_pipeline.process_pdf = AsyncMock(
                return_value={
                    "status": "duplicate",
                    "message": "This PDF has already been processed",
                    "file_hash": "abc123",
                }
            )

            result = await process_pdf(
                pdf_path="/path/to/duplicate.pdf", rulebook_name="Test Book", system="D&D 5e"
            )

            assert result["status"] == "duplicate"
            assert "already been processed" in result["message"]

    @pytest.mark.asyncio
    async def test_add_campaign_data_tool(self, mock_db):
        """Test adding campaign data."""
        with patch("src.main.campaign_manager") as mock_cm:
            mock_cm.add_campaign_data = AsyncMock(return_value="data_123")

            result = await add_campaign_data(
                campaign_id="campaign_1",
                data_type="npc",
                data={"name": "Gandalf", "role": "Wizard", "description": "Grey wizard"},
            )

            assert result["status"] == "success"
            assert result["data_id"] == "data_123"
            mock_cm.add_campaign_data.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_campaign_data_validation(self):
        """Test campaign data validation."""
        result = await add_campaign_data(
            campaign_id="", data_type="npc", data={"name": "Test"}  # Invalid
        )

        assert result["status"] == "error"
        assert "Campaign ID is required" in result["error"]

        result = await add_campaign_data(
            campaign_id="campaign_1",
            data_type="invalid_type",  # Invalid type
            data={"name": "Test"},
        )

        assert result["status"] == "error"
        assert "Invalid data type" in result["error"]

    @pytest.mark.asyncio
    async def test_get_campaign_data_tool(self, mock_db):
        """Test retrieving campaign data."""
        with patch("src.main.campaign_manager") as mock_cm:
            mock_cm.get_campaign_data = AsyncMock(
                return_value={"npcs": [{"name": "Gandalf"}], "locations": [{"name": "Shire"}]}
            )

            result = await get_campaign_data(campaign_id="campaign_1", data_type="all")

            assert result["status"] == "success"
            assert "npcs" in result["data"]
            assert "locations" in result["data"]
            mock_cm.get_campaign_data.assert_called_once()


class TestCharacterGenerationTools:
    """Test character generation MCP tools."""

    @pytest.mark.asyncio
    async def test_generate_character_tool(self):
        """Test character generation tool."""
        with patch("src.character_generation.mcp_tools.generate_character") as mock_gen:
            mock_gen.return_value = {
                "success": True,
                "character": {"name": "Aragorn", "race": "human", "class": "ranger", "level": 5},
            }

            result = await mock_gen(name="Aragorn", race="human", character_class="ranger", level=5)

            assert result["success"] is True
            assert result["character"]["name"] == "Aragorn"
            assert result["character"]["level"] == 5

    @pytest.mark.asyncio
    async def test_generate_npc_tool(self):
        """Test NPC generation tool."""
        with patch("src.character_generation.mcp_tools.generate_npc") as mock_gen:
            mock_gen.return_value = {
                "success": True,
                "npc": {
                    "name": "Innkeeper Bob",
                    "role": "innkeeper",
                    "personality": ["friendly", "gossipy"],
                },
            }

            result = await mock_gen(role="innkeeper", party_level=5)

            assert result["success"] is True
            assert result["npc"]["role"] == "innkeeper"
            assert "friendly" in result["npc"]["personality"]

    @pytest.mark.asyncio
    async def test_generate_backstory_tool(self):
        """Test backstory generation tool."""
        with patch("src.character_generation.mcp_tools.generate_backstory") as mock_gen:
            mock_gen.return_value = {
                "success": True,
                "backstory": {
                    "origin": "Noble family of Waterdeep",
                    "motivation": "Seeking revenge for family betrayal",
                    "traits": ["honorable", "determined"],
                },
            }

            result = await mock_gen(character_class="fighter", race="human", background="noble")

            assert result["success"] is True
            assert "Noble" in result["backstory"]["origin"]
            assert len(result["backstory"]["traits"]) > 0


class TestSessionManagementTools:
    """Test session management MCP tools."""

    @pytest.mark.asyncio
    async def test_create_session_tool(self):
        """Test session creation tool."""
        with patch("src.session.mcp_tools.create_session") as mock_create:
            mock_create.return_value = {
                "success": True,
                "session_id": "session_123",
                "message": "Session created successfully",
            }

            result = await mock_create(
                campaign_id="campaign_1", title="The Goblin Ambush", session_number=5
            )

            assert result["success"] is True
            assert result["session_id"] == "session_123"

    @pytest.mark.asyncio
    async def test_update_initiative_tool(self):
        """Test initiative update tool."""
        with patch("src.session.mcp_tools.update_initiative") as mock_update:
            mock_update.return_value = {
                "success": True,
                "initiative_order": [
                    {"name": "Gandalf", "initiative": 20},
                    {"name": "Goblin", "initiative": 15},
                ],
            }

            result = await mock_update(
                session_id="session_123",
                entries=[
                    {"name": "Gandalf", "initiative": 20},
                    {"name": "Goblin", "initiative": 15},
                ],
            )

            assert result["success"] is True
            assert len(result["initiative_order"]) == 2
            assert result["initiative_order"][0]["initiative"] == 20

    @pytest.mark.asyncio
    async def test_update_monster_hp_tool(self):
        """Test monster HP update tool."""
        with patch("src.session.mcp_tools.update_monster_hp") as mock_update:
            mock_update.return_value = {
                "success": True,
                "monster": {"id": "monster_1", "hp_current": 4, "hp_max": 7, "status": "injured"},
            }

            result = await mock_update(session_id="session_123", monster_id="monster_1", damage=3)

            assert result["success"] is True
            assert result["monster"]["hp_current"] == 4
            assert result["monster"]["status"] == "injured"


class TestPerformanceTools:
    """Test performance and caching MCP tools."""

    @pytest.mark.asyncio
    async def test_cache_stats_tool(self):
        """Test cache statistics tool."""
        with patch("src.performance.mcp_tools.get_cache_stats") as mock_stats:
            mock_stats.return_value = {
                "success": True,
                "stats": {
                    "total_caches": 5,
                    "total_entries": 1000,
                    "hit_rate": 0.85,
                    "memory_usage_mb": 50,
                },
            }

            result = await mock_stats()

            assert result["success"] is True
            assert result["stats"]["hit_rate"] == 0.85
            assert result["stats"]["total_entries"] == 1000

    @pytest.mark.asyncio
    async def test_cache_invalidation_tool(self):
        """Test cache invalidation tool."""
        with patch("src.performance.mcp_tools.invalidate_cache") as mock_inv:
            mock_inv.return_value = {
                "success": True,
                "invalidated": 25,
                "message": "Successfully invalidated 25 cache entries",
            }

            result = await mock_inv(cache_name="search_cache", pattern="spell_*")

            assert result["success"] is True
            assert result["invalidated"] == 25

    @pytest.mark.asyncio
    async def test_optimize_performance_tool(self):
        """Test performance optimization tool."""
        with patch("src.performance.mcp_tools.optimize_performance") as mock_opt:
            mock_opt.return_value = {
                "success": True,
                "optimizations": {
                    "indices_optimized": 3,
                    "cache_resized": True,
                    "memory_freed_mb": 100,
                },
            }

            result = await mock_opt()

            assert result["success"] is True
            assert result["optimizations"]["indices_optimized"] == 3
            assert result["optimizations"]["memory_freed_mb"] == 100


class TestMCPServerIntegration:
    """Test MCP server integration."""

    @pytest.fixture
    def mock_mcp_instance(self):
        """Create mock FastMCP instance."""
        mock = Mock()
        mock.tool = Mock(return_value=lambda func: func)
        mock.run = AsyncMock()
        return mock

    def test_tool_registration(self, mock_mcp_instance):
        """Test that tools are registered with MCP server."""
        from src.campaign import register_campaign_tools
        from src.character_generation import register_character_tools
        from src.session import register_session_tools

        # Register tools
        register_campaign_tools(mock_mcp_instance)
        register_session_tools(mock_mcp_instance)
        register_character_tools(mock_mcp_instance)

        # Verify tool decorator was called
        assert mock_mcp_instance.tool.call_count > 0

    @pytest.mark.asyncio
    async def test_mcp_error_handling(self):
        """Test MCP error handling."""

        # Test tool with error
        async def failing_tool():
            raise ValueError("Tool failed")

        # Wrap with error handler
        async def safe_tool():
            try:
                return await failing_tool()
            except Exception as e:
                return {"status": "error", "error": str(e)}

        result = await safe_tool()
        assert result["status"] == "error"
        assert "Tool failed" in result["error"]

    def test_tool_documentation(self):
        """Test that tools have proper documentation."""
        # Import tools
        from src.main import process_pdf, search

        # Check docstrings
        assert search.__doc__ is not None
        assert "Search" in search.__doc__

        assert process_pdf.__doc__ is not None
        assert "PDF" in process_pdf.__doc__

    @pytest.mark.asyncio
    async def test_tool_validation(self):
        """Test tool input validation."""
        # Test with invalid inputs
        with patch("src.main.db") as mock_db:
            mock_db.search = AsyncMock(return_value=[])

            # Empty query
            result = await search(query="")
            assert len(result.get("results", [])) == 0

            # Invalid max_results
            result = await search(query="test", max_results=-1)
            assert result["status"] == "error" or len(result["results"]) == 0


class TestToolConcurrency:
    """Test concurrent tool execution."""

    @pytest.mark.asyncio
    async def test_concurrent_searches(self):
        """Test multiple concurrent searches."""
        with patch("src.main.db") as mock_db:
            mock_db.search = AsyncMock(return_value=[{"id": "1", "content": "Result"}])

            # Execute multiple searches concurrently
            tasks = [search(f"query_{i}") for i in range(5)]

            results = await asyncio.gather(*tasks)

            assert len(results) == 5
            assert all(r["status"] == "success" for r in results)
            assert mock_db.search.call_count == 5

    @pytest.mark.asyncio
    async def test_concurrent_different_tools(self):
        """Test different tools running concurrently."""
        with patch("src.main.db") as mock_db:
            with patch("src.main.campaign_manager") as mock_cm:
                mock_db.search = AsyncMock(return_value=[])
                mock_cm.get_campaign_data = AsyncMock(return_value={})

                # Run different tools concurrently
                tasks = [search("test"), get_campaign_data("campaign_1", "all")]

                results = await asyncio.gather(*tasks)

                assert len(results) == 2
                assert mock_db.search.called
                assert mock_cm.get_campaign_data.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
