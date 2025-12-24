"""Basic tests for TTRPG Assistant MCP Server."""

from unittest.mock import MagicMock, patch

import pytest

from src.core.database import ChromaDBManager


def test_import():
    """Test that main modules can be imported."""
    import src.main

    assert src.main.mcp is not None


@patch("src.core.database.chromadb.Client")
def test_database_initialization(mock_client):
    """Test ChromaDB manager initialization."""
    mock_client.return_value = MagicMock()

    # Create a new database manager
    db = ChromaDBManager()

    # Verify client was created
    assert db.client is not None
    assert db.embedding_function is not None
    # Check for the presence of expected collection names instead of hard-coding the count
    expected_collections = [
        "rulebooks",
        "campaigns",
        "characters",
        "monsters",
        "items",
    ]  # Update as appropriate
    for name in expected_collections:
        assert name in db.collections


@pytest.mark.asyncio
async def test_server_info_tool():
    """Test the server_info MCP tool."""
    from src.main import server_info

    with patch("src.main.db") as mock_db:
        mock_db.collections = {
            "rulebooks": MagicMock(),
            "campaigns": MagicMock(),
        }
        mock_db.get_collection_stats.return_value = {
            "name": "test",
            "count": 0,
            "metadata": {},
        }

        result = await server_info()

        assert result["name"] == "TTRPG Assistant"
        assert result["version"] == "0.1.0"
        assert "settings" in result


@pytest.mark.asyncio
async def test_search_tool():
    """Test the search MCP tool."""
    from src.main import search

    with patch("src.main.db") as mock_db:
        mock_db.search.return_value = [
            {
                "content": "Fireball: A bright streak...",
                "metadata": {
                    "source": "Player's Handbook",
                    "page": 241,
                    "section": "Spells",
                },
                "distance": 0.1,
            }
        ]

        result = await search(
            query="fireball spell",
            max_results=5,
        )

        assert result["status"] == "success"
        assert len(result["results"]) == 1
        assert "Fireball" in result["results"][0]["content"]
