"""Unit tests for campaign management system."""

import uuid
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.campaign.campaign_manager import CampaignManager
from src.campaign.models import Campaign, CampaignData, LocationData, NPCData, PlotPoint
from src.campaign.rulebook_linker import RulebookLinker


class TestCampaignModels:
    """Test campaign data models."""

    def test_campaign_creation(self):
        """Test creating a campaign."""
        campaign = Campaign(
            id="test_campaign",
            name="Test Campaign",
            system="D&D 5e",
            created_at=datetime.now(),
            description="A test campaign",
        )

        assert campaign.id == "test_campaign"
        assert campaign.name == "Test Campaign"
        assert campaign.system == "D&D 5e"
        assert campaign.description == "A test campaign"

    def test_campaign_serialization(self):
        """Test campaign serialization to dict."""
        campaign = Campaign(
            id="test_campaign", name="Test Campaign", system="D&D 5e", created_at=datetime.now()
        )

        data = campaign.to_dict()

        assert data["id"] == "test_campaign"
        assert data["name"] == "Test Campaign"
        assert data["system"] == "D&D 5e"
        assert "created_at" in data

    def test_npc_data_creation(self):
        """Test creating NPC data."""
        npc = NPCData(
            id="npc_1",
            name="Gandalf",
            role="Wizard",
            description="A wise wizard",
            stats={"level": 20, "hp": 100},
            location="Shire",
        )

        assert npc.name == "Gandalf"
        assert npc.role == "Wizard"
        assert npc.stats["level"] == 20
        assert npc.location == "Shire"

    def test_location_data_creation(self):
        """Test creating location data."""
        location = LocationData(
            id="loc_1",
            name="Waterdeep",
            description="City of Splendors",
            type="city",
            npcs=["npc_1", "npc_2"],
            connected_locations=["loc_2", "loc_3"],
        )

        assert location.name == "Waterdeep"
        assert location.type == "city"
        assert len(location.npcs) == 2
        assert "loc_2" in location.connected_locations

    def test_plot_point_creation(self):
        """Test creating plot points."""
        plot = PlotPoint(
            id="plot_1",
            title="The Quest Begins",
            description="Heroes meet in a tavern",
            status="active",
            dependencies=["plot_0"],
            outcomes=["plot_2", "plot_3"],
        )

        assert plot.title == "The Quest Begins"
        assert plot.status == "active"
        assert "plot_0" in plot.dependencies
        assert len(plot.outcomes) == 2


class TestCampaignManager:
    """Test campaign manager functionality."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database."""
        db = Mock()
        db.add_document = AsyncMock()
        db.search = AsyncMock(return_value=[])
        db.update_document = AsyncMock()
        db.delete_document = AsyncMock()
        db.get_document = AsyncMock()
        return db

    @pytest.fixture
    def campaign_manager(self, mock_db):
        """Create campaign manager with mock database."""
        return CampaignManager(mock_db)

    @pytest.mark.asyncio
    async def test_create_campaign(self, campaign_manager, mock_db):
        """Test creating a new campaign."""
        campaign_id = await campaign_manager.create_campaign(
            name="Test Campaign", system="D&D 5e", description="Test description"
        )

        assert campaign_id is not None
        mock_db.add_document.assert_called_once()

        # Verify document structure
        call_args = mock_db.add_document.call_args
        assert call_args[1]["collection_name"] == "campaigns"
        assert "Test Campaign" in call_args[1]["content"]

    @pytest.mark.asyncio
    async def test_get_campaign(self, campaign_manager, mock_db):
        """Test retrieving a campaign."""
        mock_db.get_document.return_value = {
            "id": "test_id",
            "content": '{"name": "Test Campaign", "system": "D&D 5e"}',
            "metadata": {"campaign_id": "test_id"},
        }

        campaign = await campaign_manager.get_campaign("test_id")

        assert campaign is not None
        assert campaign["name"] == "Test Campaign"
        mock_db.get_document.assert_called_with("campaigns", "test_id")

    @pytest.mark.asyncio
    async def test_update_campaign(self, campaign_manager, mock_db):
        """Test updating campaign data."""
        updates = {"description": "Updated description", "status": "active"}

        result = await campaign_manager.update_campaign("test_id", updates)

        assert result is True
        mock_db.update_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_campaign(self, campaign_manager, mock_db):
        """Test deleting a campaign."""
        result = await campaign_manager.delete_campaign("test_id")

        assert result is True
        mock_db.delete_document.assert_called_with("campaigns", "test_id")

    @pytest.mark.asyncio
    async def test_add_npc(self, campaign_manager, mock_db):
        """Test adding an NPC to a campaign."""
        npc_data = {"name": "Gandalf", "role": "Wizard", "description": "Grey wizard"}

        npc_id = await campaign_manager.add_npc("test_campaign", npc_data)

        assert npc_id is not None
        mock_db.add_document.assert_called()

        # Verify NPC was added with campaign reference
        call_args = mock_db.add_document.call_args
        assert call_args[1]["metadata"]["campaign_id"] == "test_campaign"
        assert call_args[1]["metadata"]["type"] == "npc"

    @pytest.mark.asyncio
    async def test_add_location(self, campaign_manager, mock_db):
        """Test adding a location to a campaign."""
        location_data = {"name": "Waterdeep", "type": "city", "description": "City of Splendors"}

        location_id = await campaign_manager.add_location("test_campaign", location_data)

        assert location_id is not None
        mock_db.add_document.assert_called()

        # Verify location metadata
        call_args = mock_db.add_document.call_args
        assert call_args[1]["metadata"]["type"] == "location"

    @pytest.mark.asyncio
    async def test_add_plot_point(self, campaign_manager, mock_db):
        """Test adding a plot point."""
        plot_data = {"title": "The Quest", "description": "Main quest line", "status": "active"}

        plot_id = await campaign_manager.add_plot_point("test_campaign", plot_data)

        assert plot_id is not None
        mock_db.add_document.assert_called()

    @pytest.mark.asyncio
    async def test_get_campaign_npcs(self, campaign_manager, mock_db):
        """Test retrieving all NPCs for a campaign."""
        mock_db.search.return_value = [
            {"id": "npc_1", "content": '{"name": "Gandalf"}', "metadata": {"type": "npc"}},
            {"id": "npc_2", "content": '{"name": "Frodo"}', "metadata": {"type": "npc"}},
        ]

        npcs = await campaign_manager.get_campaign_npcs("test_campaign")

        assert len(npcs) == 2
        assert any(npc["name"] == "Gandalf" for npc in npcs)
        assert any(npc["name"] == "Frodo" for npc in npcs)

    @pytest.mark.asyncio
    async def test_campaign_versioning(self, campaign_manager, mock_db):
        """Test campaign version history."""
        # Create version snapshot
        version_id = await campaign_manager.create_version_snapshot(
            "test_campaign", "Before major battle"
        )

        assert version_id is not None

        # Verify version was stored
        call_args = mock_db.add_document.call_args
        assert call_args[1]["metadata"]["type"] == "version"
        assert call_args[1]["metadata"]["campaign_id"] == "test_campaign"

    @pytest.mark.asyncio
    async def test_rollback_campaign(self, campaign_manager, mock_db):
        """Test rolling back to a previous version."""
        # Mock version data
        mock_db.get_document.return_value = {
            "id": "version_1",
            "content": '{"snapshot": {"name": "Old Campaign"}}',
            "metadata": {"type": "version"},
        }

        result = await campaign_manager.rollback_to_version("test_campaign", "version_1")

        assert result is True
        mock_db.update_document.assert_called()


class TestRulebookLinker:
    """Test rulebook linking functionality."""

    @pytest.fixture
    def mock_db(self):
        """Create mock database."""
        db = Mock()
        db.search = AsyncMock(return_value=[])
        db.add_document = AsyncMock()
        return db

    @pytest.fixture
    def rulebook_linker(self, mock_db):
        """Create rulebook linker with mock database."""
        return RulebookLinker(mock_db)

    @pytest.mark.asyncio
    async def test_link_npc_to_rules(self, rulebook_linker, mock_db):
        """Test linking NPC to rulebook entries."""
        # Mock rulebook search results
        mock_db.search.return_value = [
            {
                "id": "rule_1",
                "content": "Wizard class features",
                "metadata": {"type": "class", "name": "wizard"},
            }
        ]

        npc_data = {"name": "Gandalf", "class": "wizard", "race": "human"}

        links = await rulebook_linker.link_npc_to_rules(npc_data)

        assert len(links) > 0
        assert any(link["type"] == "class" for link in links)
        mock_db.search.assert_called()

    @pytest.mark.asyncio
    async def test_link_location_to_rules(self, rulebook_linker, mock_db):
        """Test linking location to rulebook entries."""
        mock_db.search.return_value = [
            {"id": "rule_1", "content": "City rules", "metadata": {"type": "settlement"}}
        ]

        location_data = {"name": "Waterdeep", "type": "city"}

        links = await rulebook_linker.link_location_to_rules(location_data)

        assert len(links) > 0
        mock_db.search.assert_called()

    @pytest.mark.asyncio
    async def test_auto_link_campaign(self, rulebook_linker, mock_db):
        """Test automatic linking of campaign elements."""
        campaign_data = {
            "npcs": [{"id": "npc_1", "class": "wizard"}, {"id": "npc_2", "class": "fighter"}],
            "locations": [{"id": "loc_1", "type": "city"}],
        }

        # Mock search to return different results for each query
        mock_db.search.return_value = [{"id": "rule_1", "content": "Class rules"}]

        links = await rulebook_linker.auto_link_campaign(campaign_data)

        assert "npcs" in links
        assert "locations" in links
        assert len(links["npcs"]) > 0

    @pytest.mark.asyncio
    async def test_suggest_rules(self, rulebook_linker, mock_db):
        """Test suggesting relevant rules based on context."""
        context = {
            "current_scene": "combat",
            "npcs_involved": ["wizard", "goblin"],
            "location_type": "dungeon",
        }

        mock_db.search.return_value = [
            {"id": "rule_1", "content": "Combat rules"},
            {"id": "rule_2", "content": "Spell casting in combat"},
        ]

        suggestions = await rulebook_linker.suggest_relevant_rules(context)

        assert len(suggestions) > 0
        assert any("combat" in s["content"].lower() for s in suggestions)


class TestCampaignIntegration:
    """Integration tests for campaign management."""

    @pytest.mark.asyncio
    async def test_campaign_workflow(self):
        """Test complete campaign workflow."""
        mock_db = Mock()
        mock_db.add_document = AsyncMock()
        mock_db.search = AsyncMock(return_value=[])
        mock_db.get_document = AsyncMock()
        mock_db.update_document = AsyncMock()

        manager = CampaignManager(mock_db)

        # Create campaign
        campaign_id = await manager.create_campaign("Epic Campaign", "D&D 5e", "An epic adventure")

        # Add NPCs
        npc1 = await manager.add_npc(campaign_id, {"name": "Villain", "role": "antagonist"})

        # Add locations
        loc1 = await manager.add_location(campaign_id, {"name": "Dark Tower", "type": "dungeon"})

        # Add plot points
        plot1 = await manager.add_plot_point(
            campaign_id, {"title": "Final Battle", "npcs": [npc1], "location": loc1}
        )

        # Verify all elements were created
        assert campaign_id is not None
        assert npc1 is not None
        assert loc1 is not None
        assert plot1 is not None

        # Verify database calls
        assert mock_db.add_document.call_count >= 4

    @pytest.mark.asyncio
    async def test_campaign_search_integration(self):
        """Test searching within campaign context."""
        mock_db = Mock()
        mock_db.search = AsyncMock()

        manager = CampaignManager(mock_db)

        # Setup mock search results
        mock_db.search.return_value = [
            {"id": "1", "content": "Campaign element", "metadata": {"campaign_id": "test"}}
        ]

        # Search within campaign
        results = await manager.search_campaign_content("test", "dragon")

        # Verify search was scoped to campaign
        call_args = mock_db.search.call_args
        assert call_args[1]["metadata_filter"]["campaign_id"] == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
