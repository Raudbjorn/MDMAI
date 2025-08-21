"""Tests for campaign management system."""

import json
import pytest
from datetime import datetime
from src.campaign.campaign_manager import Campaign, CampaignManager, CampaignVersion


class TestCampaign:
    """Test Campaign class functionality."""
    
    def test_campaign_creation(self):
        """Test creating a new campaign."""
        campaign = Campaign(
            campaign_id="test-123",
            name="Test Campaign",
            system="D&D 5e",
            description="A test campaign",
        )
        
        assert campaign.campaign_id == "test-123"
        assert campaign.name == "Test Campaign"
        assert campaign.system == "D&D 5e"
        assert campaign.description == "A test campaign"
        assert campaign.status == "active"
        assert isinstance(campaign.characters, list)
        assert isinstance(campaign.npcs, list)
    
    def test_campaign_to_dict(self):
        """Test converting campaign to dictionary."""
        campaign = Campaign(
            campaign_id="test-123",
            name="Test Campaign",
            system="D&D 5e",
        )
        
        data = campaign.to_dict()
        
        assert data["campaign_id"] == "test-123"
        assert data["name"] == "Test Campaign"
        assert data["system"] == "D&D 5e"
        assert "created_at" in data
        assert "updated_at" in data
    
    def test_campaign_from_dict(self):
        """Test creating campaign from dictionary."""
        data = {
            "campaign_id": "test-123",
            "name": "Test Campaign",
            "system": "D&D 5e",
            "description": "Test",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "characters": [{"name": "Bob"}],
            "status": "active",
        }
        
        campaign = Campaign.from_dict(data)
        
        assert campaign.campaign_id == "test-123"
        assert campaign.name == "Test Campaign"
        assert len(campaign.characters) == 1
        assert campaign.characters[0]["name"] == "Bob"
    
    def test_campaign_versioning(self):
        """Test campaign version management."""
        campaign = Campaign(
            campaign_id="test-123",
            name="Test Campaign",
            system="D&D 5e",
        )
        
        # Create a version
        version = campaign.create_version(metadata={"reason": "test"})
        
        assert isinstance(version, CampaignVersion)
        assert version.campaign_id == "test-123"
        assert len(campaign.versions) == 1
        assert campaign.current_version == 1
        
        # Modify campaign
        campaign.name = "Modified Campaign"
        
        # Create another version
        version2 = campaign.create_version()
        assert len(campaign.versions) == 2
        assert campaign.current_version == 2
        
        # Rollback
        success = campaign.rollback_to_version(version.version_id)
        assert success == True
        assert campaign.name == "Test Campaign"  # Rolled back


class TestCampaignManager:
    """Test CampaignManager functionality."""
    
    def test_create_campaign(self):
        """Test creating a campaign through manager."""
        manager = CampaignManager(db_manager=None)
        
        campaign = manager.create_campaign(
            name="Test Campaign",
            system="D&D 5e",
            description="Test description",
        )
        
        assert campaign.name == "Test Campaign"
        assert campaign.system == "D&D 5e"
        assert campaign.campaign_id in manager.campaigns
    
    def test_get_campaign(self):
        """Test retrieving a campaign."""
        manager = CampaignManager(db_manager=None)
        
        # Create campaign
        campaign = manager.create_campaign(
            name="Test Campaign",
            system="D&D 5e",
        )
        
        # Retrieve it
        retrieved = manager.get_campaign(campaign.campaign_id)
        assert retrieved is not None
        assert retrieved.name == "Test Campaign"
        
        # Try non-existent
        not_found = manager.get_campaign("fake-id")
        assert not_found is None
    
    def test_update_campaign(self):
        """Test updating campaign data."""
        manager = CampaignManager(db_manager=None)
        
        # Create campaign
        campaign = manager.create_campaign(
            name="Test Campaign",
            system="D&D 5e",
        )
        
        # Update it
        updated = manager.update_campaign(
            campaign.campaign_id,
            {
                "name": "Updated Campaign",
                "description": "New description",
            },
        )
        
        assert updated is not None
        assert updated.name == "Updated Campaign"
        assert updated.description == "New description"
    
    def test_add_campaign_data(self):
        """Test adding data to a campaign."""
        manager = CampaignManager(db_manager=None)
        
        # Create campaign
        campaign = manager.create_campaign(
            name="Test Campaign",
            system="D&D 5e",
        )
        
        # Add character
        success = manager.add_campaign_data(
            campaign.campaign_id,
            "characters",
            {"name": "Aragorn", "class": "Ranger", "level": 10},
        )
        
        assert success == True
        
        # Check it was added
        campaign = manager.get_campaign(campaign.campaign_id)
        assert len(campaign.characters) == 1
        assert campaign.characters[0]["name"] == "Aragorn"
        
        # Add NPC
        success = manager.add_campaign_data(
            campaign.campaign_id,
            "npcs",
            {"name": "Gandalf", "role": "Wizard"},
        )
        
        assert success == True
        assert len(campaign.npcs) == 1
    
    def test_get_campaign_data(self):
        """Test retrieving campaign data."""
        manager = CampaignManager(db_manager=None)
        
        # Create campaign with data
        campaign = manager.create_campaign(
            name="Test Campaign",
            system="D&D 5e",
        )
        
        manager.add_campaign_data(
            campaign.campaign_id,
            "characters",
            {"name": "Frodo"},
        )
        
        # Get all data
        all_data = manager.get_campaign_data(campaign.campaign_id)
        assert "campaign_id" in all_data
        assert "characters" in all_data
        
        # Get specific data type
        char_data = manager.get_campaign_data(campaign.campaign_id, "characters")
        assert "characters" in char_data
        assert len(char_data["characters"]) == 1
    
    def test_list_campaigns(self):
        """Test listing campaigns."""
        manager = CampaignManager(db_manager=None)
        
        # Create multiple campaigns
        campaign1 = manager.create_campaign("Campaign 1", "D&D 5e")
        campaign2 = manager.create_campaign("Campaign 2", "Pathfinder")
        campaign3 = manager.create_campaign("Campaign 3", "D&D 5e")
        
        # List all
        all_campaigns = manager.list_campaigns()
        assert len(all_campaigns) >= 3
        
        # Filter by system
        dnd_campaigns = manager.list_campaigns(system="D&D 5e")
        assert len(dnd_campaigns) >= 2
        assert all(c.system == "D&D 5e" for c in dnd_campaigns)
    
    def test_campaign_archival(self):
        """Test archiving a campaign."""
        manager = CampaignManager(db_manager=None)
        
        # Create campaign
        campaign = manager.create_campaign(
            name="Test Campaign",
            system="D&D 5e",
        )
        
        # Archive it
        success = manager.delete_campaign(campaign.campaign_id, archive=True)
        assert success == True
        
        # Check it's archived
        campaign = manager.get_campaign(campaign.campaign_id)
        assert campaign.status == "archived"
        
        # Should not appear in active list
        active = manager.list_campaigns(status="active")
        assert campaign.campaign_id not in [c.campaign_id for c in active]
    
    def test_rulebook_links(self):
        """Test adding and managing rulebook links."""
        manager = CampaignManager(db_manager=None)
        
        # Create campaign
        campaign = manager.create_campaign(
            name="Test Campaign",
            system="D&D 5e",
        )
        
        # Add rulebook link
        success = manager.add_rulebook_link(
            campaign.campaign_id,
            "phb-123",
            "reference",
            {"page": 42},
        )
        
        assert success == True
        
        # Get links
        links = manager.get_rulebook_links(campaign.campaign_id)
        assert len(links) == 1
        assert links[0]["rulebook_id"] == "phb-123"
        assert links[0]["link_type"] == "reference"
    
    def test_search_campaign_data(self):
        """Test searching within campaign data."""
        manager = CampaignManager(db_manager=None)
        
        # Create campaign with data
        campaign = manager.create_campaign(
            name="Test Campaign",
            system="D&D 5e",
        )
        
        manager.add_campaign_data(
            campaign.campaign_id,
            "characters",
            {"name": "Aragorn", "class": "Ranger"},
        )
        
        manager.add_campaign_data(
            campaign.campaign_id,
            "npcs",
            {"name": "Boromir", "role": "Fighter"},
        )
        
        manager.add_campaign_data(
            campaign.campaign_id,
            "locations",
            {"name": "Moria", "description": "Ancient dwarven city"},
        )
        
        # Search for "aragorn"
        results = manager.search_campaign_data(
            campaign.campaign_id,
            "aragorn",
        )
        
        assert len(results) == 1
        assert results[0]["type"] == "character"
        assert results[0]["data"]["name"] == "Aragorn"
        
        # Search for "moria"
        results = manager.search_campaign_data(
            campaign.campaign_id,
            "moria",
        )
        
        assert len(results) == 1
        assert results[0]["type"] == "location"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])