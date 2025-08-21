"""Campaign management system for TTRPG Assistant."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import copy

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class CampaignVersion:
    """Represents a version of campaign data for rollback functionality."""
    
    def __init__(self, version_id: str, campaign_id: str, data: Dict[str, Any], metadata: Dict[str, Any] = None):
        """
        Initialize campaign version.
        
        Args:
            version_id: Unique version identifier
            campaign_id: Campaign identifier
            data: Campaign data snapshot
            metadata: Version metadata
        """
        self.version_id = version_id
        self.campaign_id = campaign_id
        self.data = copy.deepcopy(data)
        self.created_at = datetime.utcnow()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert version to dictionary."""
        return {
            "version_id": self.version_id,
            "campaign_id": self.campaign_id,
            "data": self.data,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


class Campaign:
    """Represents a TTRPG campaign with all associated data."""
    
    def __init__(
        self,
        campaign_id: str,
        name: str,
        system: str,
        description: str = "",
        created_at: Optional[datetime] = None,
    ):
        """
        Initialize campaign.
        
        Args:
            campaign_id: Unique campaign identifier
            name: Campaign name
            system: Game system
            description: Campaign description
            created_at: Creation timestamp
        """
        self.campaign_id = campaign_id
        self.name = name
        self.system = system
        self.description = description
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        # Campaign data containers
        self.characters: List[Dict[str, Any]] = []
        self.npcs: List[Dict[str, Any]] = []
        self.locations: List[Dict[str, Any]] = []
        self.plot_points: List[Dict[str, Any]] = []
        self.sessions: List[Dict[str, Any]] = []
        self.notes: List[Dict[str, Any]] = []
        self.custom_data: Dict[str, Any] = {}
        
        # Rulebook linkages
        self.rulebook_links: List[Dict[str, Any]] = []
        
        # Campaign settings
        self.settings: Dict[str, Any] = {
            "allow_homebrew": True,
            "experience_system": "standard",
            "difficulty": "normal",
        }
        
        # Version history
        self.versions: List[CampaignVersion] = []
        self.current_version: int = 0
        
        # Campaign status
        self.status = "active"  # active, paused, completed, archived
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert campaign to dictionary."""
        return {
            "campaign_id": self.campaign_id,
            "name": self.name,
            "system": self.system,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "characters": self.characters,
            "npcs": self.npcs,
            "locations": self.locations,
            "plot_points": self.plot_points,
            "sessions": self.sessions,
            "notes": self.notes,
            "custom_data": self.custom_data,
            "rulebook_links": self.rulebook_links,
            "settings": self.settings,
            "status": self.status,
            "current_version": self.current_version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Campaign":
        """Create campaign from dictionary."""
        campaign = cls(
            campaign_id=data["campaign_id"],
            name=data["name"],
            system=data["system"],
            description=data.get("description", ""),
        )
        
        # Restore timestamps
        if "created_at" in data:
            campaign.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            campaign.updated_at = datetime.fromisoformat(data["updated_at"])
        
        # Restore data
        campaign.characters = data.get("characters", [])
        campaign.npcs = data.get("npcs", [])
        campaign.locations = data.get("locations", [])
        campaign.plot_points = data.get("plot_points", [])
        campaign.sessions = data.get("sessions", [])
        campaign.notes = data.get("notes", [])
        campaign.custom_data = data.get("custom_data", {})
        campaign.rulebook_links = data.get("rulebook_links", [])
        campaign.settings = data.get("settings", campaign.settings)
        campaign.status = data.get("status", "active")
        campaign.current_version = data.get("current_version", 0)
        
        return campaign
    
    def create_version(self, metadata: Dict[str, Any] = None) -> CampaignVersion:
        """
        Create a new version snapshot.
        
        Args:
            metadata: Version metadata
            
        Returns:
            Created version
        """
        version = CampaignVersion(
            version_id=str(uuid.uuid4()),
            campaign_id=self.campaign_id,
            data=self.to_dict(),
            metadata=metadata,
        )
        
        self.versions.append(version)
        self.current_version += 1
        
        # Keep only last N versions to save space
        max_versions = 10
        if len(self.versions) > max_versions:
            self.versions = self.versions[-max_versions:]
        
        return version
    
    def rollback_to_version(self, version_id: str) -> bool:
        """
        Rollback to a specific version.
        
        Args:
            version_id: Version identifier
            
        Returns:
            True if successful
        """
        for version in self.versions:
            if version.version_id == version_id:
                # Restore data from version
                data = version.data
                self.name = data["name"]
                self.system = data["system"]
                self.description = data["description"]
                self.characters = data["characters"]
                self.npcs = data["npcs"]
                self.locations = data["locations"]
                self.plot_points = data["plot_points"]
                self.sessions = data["sessions"]
                self.notes = data["notes"]
                self.custom_data = data["custom_data"]
                self.rulebook_links = data["rulebook_links"]
                self.settings = data["settings"]
                self.updated_at = datetime.utcnow()
                
                return True
        
        return False


class CampaignManager:
    """Manages campaigns and their data."""
    
    def __init__(self, db_manager=None):
        """
        Initialize campaign manager.
        
        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager
        self.campaigns: Dict[str, Campaign] = {}
        self.campaign_cache_dir = settings.cache_dir / "campaigns"
        self.campaign_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load cached campaigns
        self._load_cached_campaigns()
    
    def create_campaign(
        self,
        name: str,
        system: str,
        description: str = "",
        settings: Optional[Dict[str, Any]] = None,
    ) -> Campaign:
        """
        Create a new campaign.
        
        Args:
            name: Campaign name
            system: Game system
            description: Campaign description
            settings: Campaign settings
            
        Returns:
            Created campaign
        """
        campaign_id = str(uuid.uuid4())
        
        campaign = Campaign(
            campaign_id=campaign_id,
            name=name,
            system=system,
            description=description,
        )
        
        if settings:
            campaign.settings.update(settings)
        
        # Store in memory
        self.campaigns[campaign_id] = campaign
        
        # Save to cache
        self._save_campaign(campaign)
        
        # Store in database if available
        if self.db:
            self._store_in_database(campaign)
        
        logger.info(f"Campaign created", campaign_id=campaign_id, name=name)
        
        return campaign
    
    def get_campaign(self, campaign_id: str) -> Optional[Campaign]:
        """
        Get a campaign by ID.
        
        Args:
            campaign_id: Campaign identifier
            
        Returns:
            Campaign or None
        """
        # Check memory cache
        if campaign_id in self.campaigns:
            return self.campaigns[campaign_id]
        
        # Try loading from database
        if self.db:
            campaign_data = self._load_from_database(campaign_id)
            if campaign_data:
                campaign = Campaign.from_dict(campaign_data)
                self.campaigns[campaign_id] = campaign
                return campaign
        
        return None
    
    def update_campaign(
        self,
        campaign_id: str,
        updates: Dict[str, Any],
        create_version: bool = True,
    ) -> Optional[Campaign]:
        """
        Update campaign data.
        
        Args:
            campaign_id: Campaign identifier
            updates: Update dictionary
            create_version: Whether to create a version snapshot
            
        Returns:
            Updated campaign or None
        """
        campaign = self.get_campaign(campaign_id)
        if not campaign:
            logger.warning(f"Campaign not found", campaign_id=campaign_id)
            return None
        
        # Create version snapshot if requested
        if create_version:
            campaign.create_version(metadata={"update_type": "manual_update"})
        
        # Apply updates
        for key, value in updates.items():
            if key == "name":
                campaign.name = value
            elif key == "description":
                campaign.description = value
            elif key == "settings":
                campaign.settings.update(value)
            elif key == "status":
                campaign.status = value
            elif key in ["characters", "npcs", "locations", "plot_points", "sessions", "notes"]:
                # Handle list updates
                current_list = getattr(campaign, key)
                if isinstance(value, list):
                    setattr(campaign, key, value)
                elif isinstance(value, dict):
                    # Single item update
                    current_list.append(value)
            elif key == "custom_data":
                campaign.custom_data.update(value)
        
        campaign.updated_at = datetime.utcnow()
        
        # Save changes
        self._save_campaign(campaign)
        if self.db:
            self._store_in_database(campaign)
        
        logger.info(f"Campaign updated", campaign_id=campaign_id)
        
        return campaign
    
    def add_campaign_data(
        self,
        campaign_id: str,
        data_type: str,
        data: Dict[str, Any],
    ) -> bool:
        """
        Add data to a campaign.
        
        Args:
            campaign_id: Campaign identifier
            data_type: Type of data (characters, npcs, etc.)
            data: Data to add
            
        Returns:
            True if successful
        """
        campaign = self.get_campaign(campaign_id)
        if not campaign:
            return False
        
        # Add unique ID if not present
        if "id" not in data:
            data["id"] = str(uuid.uuid4())
        
        # Add timestamp
        data["created_at"] = datetime.utcnow().isoformat()
        
        # Add to appropriate list
        if data_type == "characters":
            campaign.characters.append(data)
        elif data_type == "npcs":
            campaign.npcs.append(data)
        elif data_type == "locations":
            campaign.locations.append(data)
        elif data_type == "plot_points":
            campaign.plot_points.append(data)
        elif data_type == "sessions":
            campaign.sessions.append(data)
        elif data_type == "notes":
            campaign.notes.append(data)
        elif data_type == "custom":
            if "custom_lists" not in campaign.custom_data:
                campaign.custom_data["custom_lists"] = {}
            custom_type = data.get("custom_type", "misc")
            if custom_type not in campaign.custom_data["custom_lists"]:
                campaign.custom_data["custom_lists"][custom_type] = []
            campaign.custom_data["custom_lists"][custom_type].append(data)
        else:
            logger.warning(f"Unknown data type", data_type=data_type)
            return False
        
        campaign.updated_at = datetime.utcnow()
        
        # Save changes
        self._save_campaign(campaign)
        if self.db:
            self._store_in_database(campaign)
        
        return True
    
    def get_campaign_data(
        self,
        campaign_id: str,
        data_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get campaign data.
        
        Args:
            campaign_id: Campaign identifier
            data_type: Optional data type filter
            
        Returns:
            Campaign data dictionary
        """
        campaign = self.get_campaign(campaign_id)
        if not campaign:
            return {}
        
        if data_type:
            # Return specific data type
            if data_type == "characters":
                return {"characters": campaign.characters}
            elif data_type == "npcs":
                return {"npcs": campaign.npcs}
            elif data_type == "locations":
                return {"locations": campaign.locations}
            elif data_type == "plot_points":
                return {"plot_points": campaign.plot_points}
            elif data_type == "sessions":
                return {"sessions": campaign.sessions}
            elif data_type == "notes":
                return {"notes": campaign.notes}
            elif data_type == "settings":
                return {"settings": campaign.settings}
            elif data_type == "custom":
                return {"custom_data": campaign.custom_data}
            else:
                return {}
        
        # Return all data
        return campaign.to_dict()
    
    def delete_campaign(self, campaign_id: str, archive: bool = True) -> bool:
        """
        Delete or archive a campaign.
        
        Args:
            campaign_id: Campaign identifier
            archive: Whether to archive instead of delete
            
        Returns:
            True if successful
        """
        campaign = self.get_campaign(campaign_id)
        if not campaign:
            return False
        
        if archive:
            # Archive the campaign
            campaign.status = "archived"
            campaign.updated_at = datetime.utcnow()
            self._save_campaign(campaign)
            
            # Move to archive in database
            if self.db:
                self._archive_in_database(campaign)
            
            logger.info(f"Campaign archived", campaign_id=campaign_id)
        else:
            # Delete from memory
            if campaign_id in self.campaigns:
                del self.campaigns[campaign_id]
            
            # Delete from cache
            cache_file = self.campaign_cache_dir / f"{campaign_id}.json"
            if cache_file.exists():
                cache_file.unlink()
            
            # Delete from database
            if self.db:
                self._delete_from_database(campaign_id)
            
            logger.info(f"Campaign deleted", campaign_id=campaign_id)
        
        return True
    
    def list_campaigns(
        self,
        system: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Campaign]:
        """
        List campaigns with optional filters.
        
        Args:
            system: Filter by game system
            status: Filter by status
            
        Returns:
            List of campaigns
        """
        campaigns = list(self.campaigns.values())
        
        # Apply filters
        if system:
            campaigns = [c for c in campaigns if c.system == system]
        if status:
            campaigns = [c for c in campaigns if c.status == status]
        
        # Sort by update time
        campaigns.sort(key=lambda c: c.updated_at, reverse=True)
        
        return campaigns
    
    def add_rulebook_link(
        self,
        campaign_id: str,
        rulebook_id: str,
        link_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add a rulebook link to campaign.
        
        Args:
            campaign_id: Campaign identifier
            rulebook_id: Rulebook identifier
            link_type: Type of link (reference, requirement, etc.)
            metadata: Link metadata
            
        Returns:
            True if successful
        """
        campaign = self.get_campaign(campaign_id)
        if not campaign:
            return False
        
        link = {
            "rulebook_id": rulebook_id,
            "link_type": link_type,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        
        campaign.rulebook_links.append(link)
        campaign.updated_at = datetime.utcnow()
        
        # Save changes
        self._save_campaign(campaign)
        if self.db:
            self._store_in_database(campaign)
        
        return True
    
    def get_rulebook_links(self, campaign_id: str) -> List[Dict[str, Any]]:
        """
        Get rulebook links for a campaign.
        
        Args:
            campaign_id: Campaign identifier
            
        Returns:
            List of rulebook links
        """
        campaign = self.get_campaign(campaign_id)
        if not campaign:
            return []
        
        return campaign.rulebook_links
    
    def validate_links(self, campaign_id: str) -> Dict[str, Any]:
        """
        Validate rulebook links for a campaign.
        
        Args:
            campaign_id: Campaign identifier
            
        Returns:
            Validation results
        """
        campaign = self.get_campaign(campaign_id)
        if not campaign:
            return {"valid": False, "error": "Campaign not found"}
        
        results = {
            "valid": True,
            "total_links": len(campaign.rulebook_links),
            "valid_links": [],
            "broken_links": [],
        }
        
        for link in campaign.rulebook_links:
            # Check if rulebook exists in database
            if self.db:
                rulebook_exists = self._check_rulebook_exists(link["rulebook_id"])
                if rulebook_exists:
                    results["valid_links"].append(link)
                else:
                    results["broken_links"].append(link)
                    results["valid"] = False
        
        return results
    
    def search_campaign_data(
        self,
        campaign_id: str,
        query: str,
        data_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search within campaign data.
        
        Args:
            campaign_id: Campaign identifier
            query: Search query
            data_types: Optional data types to search
            
        Returns:
            Search results
        """
        campaign = self.get_campaign(campaign_id)
        if not campaign:
            return []
        
        results = []
        query_lower = query.lower()
        
        # Define searchable data types
        if not data_types:
            data_types = ["characters", "npcs", "locations", "plot_points", "notes"]
        
        # Search each data type
        for data_type in data_types:
            if data_type == "characters":
                for char in campaign.characters:
                    if self._search_in_dict(char, query_lower):
                        results.append({
                            "type": "character",
                            "data": char,
                            "campaign_id": campaign_id,
                        })
            
            elif data_type == "npcs":
                for npc in campaign.npcs:
                    if self._search_in_dict(npc, query_lower):
                        results.append({
                            "type": "npc",
                            "data": npc,
                            "campaign_id": campaign_id,
                        })
            
            elif data_type == "locations":
                for loc in campaign.locations:
                    if self._search_in_dict(loc, query_lower):
                        results.append({
                            "type": "location",
                            "data": loc,
                            "campaign_id": campaign_id,
                        })
            
            elif data_type == "plot_points":
                for plot in campaign.plot_points:
                    if self._search_in_dict(plot, query_lower):
                        results.append({
                            "type": "plot_point",
                            "data": plot,
                            "campaign_id": campaign_id,
                        })
            
            elif data_type == "notes":
                for note in campaign.notes:
                    if self._search_in_dict(note, query_lower):
                        results.append({
                            "type": "note",
                            "data": note,
                            "campaign_id": campaign_id,
                        })
        
        return results
    
    def _search_in_dict(self, data: Dict[str, Any], query: str) -> bool:
        """
        Search for query in dictionary values.
        
        Args:
            data: Dictionary to search
            query: Search query
            
        Returns:
            True if found
        """
        for value in data.values():
            if isinstance(value, str) and query in value.lower():
                return True
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and query in item.lower():
                        return True
            elif isinstance(value, dict):
                if self._search_in_dict(value, query):
                    return True
        
        return False
    
    def _save_campaign(self, campaign: Campaign):
        """Save campaign to cache."""
        try:
            cache_file = self.campaign_cache_dir / f"{campaign.campaign_id}.json"
            with open(cache_file, "w") as f:
                json.dump(campaign.to_dict(), f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save campaign", error=str(e))
    
    def _load_cached_campaigns(self):
        """Load campaigns from cache."""
        try:
            for cache_file in self.campaign_cache_dir.glob("*.json"):
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    campaign = Campaign.from_dict(data)
                    self.campaigns[campaign.campaign_id] = campaign
            
            logger.info(f"Loaded {len(self.campaigns)} campaigns from cache")
            
        except Exception as e:
            logger.error(f"Failed to load campaigns", error=str(e))
    
    def _store_in_database(self, campaign: Campaign):
        """Store campaign in database."""
        if not self.db:
            return
        
        try:
            self.db.add_document(
                collection_name="campaigns",
                document_id=campaign.campaign_id,
                content=json.dumps(campaign.to_dict()),
                metadata={
                    "campaign_id": campaign.campaign_id,
                    "name": campaign.name,
                    "system": campaign.system,
                    "status": campaign.status,
                    "data_type": "campaign",
                },
            )
        except Exception as e:
            logger.error(f"Failed to store campaign in database", error=str(e))
    
    def _load_from_database(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        """Load campaign from database."""
        if not self.db:
            return None
        
        try:
            doc = self.db.get_document(
                collection_name="campaigns",
                document_id=campaign_id,
            )
            
            if doc and doc.get("content"):
                return json.loads(doc["content"])
            
        except Exception as e:
            logger.error(f"Failed to load campaign from database", error=str(e))
        
        return None
    
    def _delete_from_database(self, campaign_id: str):
        """Delete campaign from database."""
        if not self.db:
            return
        
        try:
            self.db.delete_document(
                collection_name="campaigns",
                document_id=campaign_id,
            )
        except Exception as e:
            logger.error(f"Failed to delete campaign from database", error=str(e))
    
    def _archive_in_database(self, campaign: Campaign):
        """Archive campaign in database."""
        # Just update the metadata to reflect archived status
        self._store_in_database(campaign)
    
    def _check_rulebook_exists(self, rulebook_id: str) -> bool:
        """Check if rulebook exists in database."""
        if not self.db:
            return False
        
        try:
            # Check in rulebooks collection
            results = self.db.search(
                collection_name="rulebooks",
                query="",
                metadata_filter={"source_id": rulebook_id},
                max_results=1,
            )
            
            return len(results) > 0
            
        except Exception as e:
            logger.error(f"Failed to check rulebook existence", error=str(e))
            return False