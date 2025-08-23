"""Campaign management system for TTRPG Assistant."""

import json
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from config.logging_config import get_logger
from src.campaign.models import NPC, Campaign, CampaignVersion, Character, Location, PlotPoint
from src.core.database import ChromaDBManager
from src.search.error_handler import DatabaseError, handle_search_errors

logger = get_logger(__name__)


class CampaignManager:
    """Manages campaign data and operations."""

    def __init__(self, db_manager: ChromaDBManager):
        """
        Initialize campaign manager.

        Args:
            db_manager: ChromaDB manager instance
        """
        self.db_manager = db_manager
        self.active_campaign_id: Optional[str] = None
        self._campaign_cache: Dict[str, Campaign] = {}
        self._version_cache: Dict[str, List[CampaignVersion]] = {}

    @handle_search_errors()
    async def create_campaign(
        self, name: str, system: str, description: str = "", setting: str = "", **kwargs
    ) -> Dict[str, Any]:
        """
        Create a new campaign.

        Args:
            name: Campaign name
            system: Game system (e.g., "D&D 5e")
            description: Campaign description
            setting: Campaign setting
            **kwargs: Additional campaign attributes

        Returns:
            Campaign creation result
        """
        try:
            # Create campaign object
            campaign = Campaign(
                name=name, system=system, description=description, setting=setting, **kwargs
            )

            # Store in database
            campaign_data = campaign.to_dict()
            self.db_manager.add_document(
                collection_name="campaigns",
                document_id=campaign.id,
                content=json.dumps(campaign_data),
                metadata={
                    "type": "campaign",
                    "name": name,
                    "system": system,
                    "created_at": campaign.created_at.isoformat(),
                    "updated_at": campaign.updated_at.isoformat(),
                },
            )

            # Create initial version
            self._create_version(campaign, "Initial campaign creation")

            # Cache the campaign
            self._campaign_cache[campaign.id] = campaign

            logger.info(f"Campaign created: {campaign.id} - {name}")

            return {
                "success": True,
                "campaign_id": campaign.id,
                "message": f"Campaign '{name}' created successfully",
                "campaign": campaign_data,
            }

        except Exception as e:
            logger.error(f"Failed to create campaign: {str(e)}")
            raise DatabaseError(f"Failed to create campaign: {str(e)}")

    @handle_search_errors()
    async def get_campaign(self, campaign_id: str) -> Optional[Campaign]:
        """
        Get a campaign by ID.

        Args:
            campaign_id: Campaign ID

        Returns:
            Campaign object or None
        """
        # Check cache first
        if campaign_id in self._campaign_cache:
            return self._campaign_cache[campaign_id]

        try:
            # Get from database
            doc = self.db_manager.get_document("campaigns", campaign_id)
            if doc:
                campaign_data = json.loads(doc["content"])
                campaign = Campaign.from_dict(campaign_data)

                # Cache it
                self._campaign_cache[campaign_id] = campaign
                return campaign

            return None

        except Exception as e:
            logger.error(f"Failed to get campaign {campaign_id}: {str(e)}")
            raise DatabaseError(f"Failed to get campaign: {str(e)}")

    @handle_search_errors()
    async def update_campaign(
        self,
        campaign_id: str,
        updates: Dict[str, Any],
        change_description: str = "Campaign updated",
    ) -> Dict[str, Any]:
        """
        Update campaign data.

        Args:
            campaign_id: Campaign ID
            updates: Dictionary of updates
            change_description: Description of changes

        Returns:
            Update result
        """
        try:
            # Get current campaign
            campaign = await self.get_campaign(campaign_id)
            if not campaign:
                return {"success": False, "message": f"Campaign {campaign_id} not found"}

            # Create version before update
            self._create_version(campaign, change_description)

            # Apply updates
            for key, value in updates.items():
                if hasattr(campaign, key):
                    setattr(campaign, key, value)

            campaign.updated_at = datetime.utcnow()

            # Update in database
            campaign_data = campaign.to_dict()
            self.db_manager.update_document(
                collection_name="campaigns",
                document_id=campaign_id,
                content=json.dumps(campaign_data),
                metadata={
                    "type": "campaign",
                    "name": campaign.name,
                    "system": campaign.system,
                    "created_at": campaign.created_at.isoformat(),
                    "updated_at": campaign.updated_at.isoformat(),
                },
            )

            # Update cache
            self._campaign_cache[campaign_id] = campaign

            logger.info(f"Campaign updated: {campaign_id}")

            return {
                "success": True,
                "message": "Campaign updated successfully",
                "campaign": campaign_data,
            }

        except Exception as e:
            logger.error(f"Failed to update campaign {campaign_id}: {str(e)}")
            raise DatabaseError(f"Failed to update campaign: {str(e)}")

    @handle_search_errors()
    async def add_character(
        self, campaign_id: str, character_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Add a character to a campaign.

        Args:
            campaign_id: Campaign ID
            character_data: Character data

        Returns:
            Addition result
        """
        try:
            campaign = await self.get_campaign(campaign_id)
            if not campaign:
                return {"success": False, "message": f"Campaign {campaign_id} not found"}

            # Create version before modification
            self._create_version(
                campaign, f"Adding character: {character_data.get('name', 'Unknown')}"
            )

            # Create and add character
            character = Character(**character_data)
            campaign.characters.append(character)
            campaign.updated_at = datetime.utcnow()

            # Save directly to database (avoid double fetch)
            campaign_data = campaign.to_dict()
            self.db_manager.update_document(
                collection_name="campaigns",
                document_id=campaign_id,
                content=json.dumps(campaign_data),
                metadata={
                    "type": "campaign",
                    "name": campaign.name,
                    "system": campaign.system,
                    "created_at": campaign.created_at.isoformat(),
                    "updated_at": campaign.updated_at.isoformat(),
                },
            )

            # Update cache
            self._campaign_cache[campaign_id] = campaign

            logger.info(f"Character added to campaign {campaign_id}: {character.id}")

            return {
                "success": True,
                "message": f"Character '{character.name}' added successfully",
                "character_id": character.id,
                "character": character.to_dict(),
                "campaign": campaign_data,
            }

        except Exception as e:
            logger.error(f"Failed to add character: {str(e)}")
            raise DatabaseError(f"Failed to add character: {str(e)}")

    @handle_search_errors()
    async def add_npc(self, campaign_id: str, npc_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add an NPC to a campaign.

        Args:
            campaign_id: Campaign ID
            npc_data: NPC data

        Returns:
            Addition result
        """
        try:
            campaign = await self.get_campaign(campaign_id)
            if not campaign:
                return {"success": False, "message": f"Campaign {campaign_id} not found"}

            # Create version before modification
            self._create_version(campaign, f"Adding NPC: {npc_data.get('name', 'Unknown')}")

            # Create and add NPC
            npc = NPC(**npc_data)
            campaign.npcs.append(npc)
            campaign.updated_at = datetime.utcnow()

            # Save directly to database (avoid double fetch)
            campaign_data = campaign.to_dict()
            self.db_manager.update_document(
                collection_name="campaigns",
                document_id=campaign_id,
                content=json.dumps(campaign_data),
                metadata={
                    "type": "campaign",
                    "name": campaign.name,
                    "system": campaign.system,
                    "created_at": campaign.created_at.isoformat(),
                    "updated_at": campaign.updated_at.isoformat(),
                },
            )

            # Update cache
            self._campaign_cache[campaign_id] = campaign

            logger.info(f"NPC added to campaign {campaign_id}: {npc.id}")

            return {
                "success": True,
                "message": f"NPC '{npc.name}' added successfully",
                "npc_id": npc.id,
                "npc": npc.to_dict(),
                "campaign": campaign_data,
            }

        except Exception as e:
            logger.error(f"Failed to add NPC: {str(e)}")
            raise DatabaseError(f"Failed to add NPC: {str(e)}")

    @handle_search_errors()
    async def add_location(self, campaign_id: str, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a location to a campaign.

        Args:
            campaign_id: Campaign ID
            location_data: Location data

        Returns:
            Addition result
        """
        try:
            campaign = await self.get_campaign(campaign_id)
            if not campaign:
                return {"success": False, "message": f"Campaign {campaign_id} not found"}

            # Create version before modification
            self._create_version(
                campaign, f"Adding location: {location_data.get('name', 'Unknown')}"
            )

            # Create and add location
            location = Location(**location_data)
            campaign.locations.append(location)
            campaign.updated_at = datetime.utcnow()

            # Save directly to database (avoid double fetch)
            campaign_data = campaign.to_dict()
            self.db_manager.update_document(
                collection_name="campaigns",
                document_id=campaign_id,
                content=json.dumps(campaign_data),
                metadata={
                    "type": "campaign",
                    "name": campaign.name,
                    "system": campaign.system,
                    "created_at": campaign.created_at.isoformat(),
                    "updated_at": campaign.updated_at.isoformat(),
                },
            )

            # Update cache
            self._campaign_cache[campaign_id] = campaign

            logger.info(f"Location added to campaign {campaign_id}: {location.id}")

            return {
                "success": True,
                "message": f"Location '{location.name}' added successfully",
                "location_id": location.id,
                "location": location.to_dict(),
                "campaign": campaign_data,
            }

        except Exception as e:
            logger.error(f"Failed to add location: {str(e)}")
            raise DatabaseError(f"Failed to add location: {str(e)}")

    @handle_search_errors()
    async def add_plot_point(self, campaign_id: str, plot_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a plot point to a campaign.

        Args:
            campaign_id: Campaign ID
            plot_data: Plot point data

        Returns:
            Addition result
        """
        try:
            campaign = await self.get_campaign(campaign_id)
            if not campaign:
                return {"success": False, "message": f"Campaign {campaign_id} not found"}

            # Create version before modification
            self._create_version(
                campaign, f"Adding plot point: {plot_data.get('title', 'Unknown')}"
            )

            # Create and add plot point
            plot_point = PlotPoint(**plot_data)
            campaign.plot_points.append(plot_point)
            campaign.updated_at = datetime.utcnow()

            # Save directly to database (avoid double fetch)
            campaign_data = campaign.to_dict()
            self.db_manager.update_document(
                collection_name="campaigns",
                document_id=campaign_id,
                content=json.dumps(campaign_data),
                metadata={
                    "type": "campaign",
                    "name": campaign.name,
                    "system": campaign.system,
                    "created_at": campaign.created_at.isoformat(),
                    "updated_at": campaign.updated_at.isoformat(),
                },
            )

            # Update cache
            self._campaign_cache[campaign_id] = campaign

            logger.info(f"Plot point added to campaign {campaign_id}: {plot_point.id}")

            return {
                "success": True,
                "message": f"Plot point '{plot_point.title}' added successfully",
                "plot_point_id": plot_point.id,
                "plot_point": plot_point.to_dict(),
                "campaign": campaign_data,
            }

        except Exception as e:
            logger.error(f"Failed to add plot point: {str(e)}")
            raise DatabaseError(f"Failed to add plot point: {str(e)}")

    @handle_search_errors()
    async def get_campaign_data(
        self, campaign_id: str, data_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get campaign data, optionally filtered by type.

        Args:
            campaign_id: Campaign ID
            data_type: Type of data to retrieve (characters, npcs, locations, plot_points)

        Returns:
            Campaign data
        """
        try:
            campaign = await self.get_campaign(campaign_id)
            if not campaign:
                return {"success": False, "message": f"Campaign {campaign_id} not found"}

            if data_type:
                if data_type == "characters":
                    data = [c.to_dict() for c in campaign.characters]
                elif data_type == "npcs":
                    data = [n.to_dict() for n in campaign.npcs]
                elif data_type == "locations":
                    data = [l.to_dict() for l in campaign.locations]
                elif data_type == "plot_points":
                    data = [p.to_dict() for p in campaign.plot_points]
                else:
                    return {"success": False, "message": f"Unknown data type: {data_type}"}

                return {"success": True, "data_type": data_type, "data": data, "count": len(data)}
            else:
                return {"success": True, "campaign": campaign.to_dict()}

        except Exception as e:
            logger.error(f"Failed to get campaign data: {str(e)}")
            raise DatabaseError(f"Failed to get campaign data: {str(e)}")

    @handle_search_errors()
    async def list_campaigns(
        self, system: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List all campaigns.

        Args:
            system: Filter by game system
            limit: Maximum number of campaigns to return

        Returns:
            List of campaign summaries
        """
        try:
            metadata_filter = {"type": "campaign"}
            if system:
                metadata_filter["system"] = system

            docs = self.db_manager.list_documents(
                "campaigns", limit=limit, metadata_filter=metadata_filter
            )

            campaigns = []
            for doc in docs:
                if doc["metadata"]:
                    campaigns.append(
                        {
                            "id": doc["id"],
                            "name": doc["metadata"].get("name", "Unknown"),
                            "system": doc["metadata"].get("system", "Unknown"),
                            "created_at": doc["metadata"].get("created_at"),
                            "updated_at": doc["metadata"].get("updated_at"),
                        }
                    )

            return campaigns

        except Exception as e:
            logger.error(f"Failed to list campaigns: {str(e)}")
            raise DatabaseError(f"Failed to list campaigns: {str(e)}")

    @handle_search_errors()
    async def delete_campaign(self, campaign_id: str) -> Dict[str, Any]:
        """
        Delete a campaign (marks as archived, doesn't actually delete).

        Args:
            campaign_id: Campaign ID

        Returns:
            Deletion result
        """
        try:
            campaign = await self.get_campaign(campaign_id)
            if not campaign:
                return {"success": False, "message": f"Campaign {campaign_id} not found"}

            # Create version before archiving
            self._create_version(campaign, "Campaign archived")

            # Archive the campaign
            campaign.archived = True
            campaign.archived_at = datetime.utcnow()
            campaign.updated_at = datetime.utcnow()

            # Save directly to database
            campaign_data = campaign.to_dict()
            self.db_manager.update_document(
                collection_name="campaigns",
                document_id=campaign_id,
                content=json.dumps(campaign_data),
                metadata={
                    "type": "campaign",
                    "name": campaign.name,
                    "system": campaign.system,
                    "created_at": campaign.created_at.isoformat(),
                    "updated_at": campaign.updated_at.isoformat(),
                    "archived": True,
                    "archived_at": campaign.archived_at.isoformat(),
                },
            )

            # Remove from cache
            if campaign_id in self._campaign_cache:
                del self._campaign_cache[campaign_id]

            logger.info(f"Campaign archived: {campaign_id}")

            return {
                "success": True,
                "message": f"Campaign '{campaign.name}' archived successfully",
                "campaign": campaign_data,
            }

        except Exception as e:
            logger.error(f"Failed to delete campaign: {str(e)}")
            raise DatabaseError(f"Failed to delete campaign: {str(e)}")

    def _create_version(self, campaign: Campaign, description: str, retry_count: int = 0) -> bool:
        """
        Create a version snapshot of the campaign.

        Args:
            campaign: Campaign to version
            description: Change description
            retry_count: Current retry attempt

        Returns:
            True if version was created successfully, False otherwise
        """
        MAX_RETRIES = 3

        try:
            # Get current version number
            version_number = 1
            if campaign.id in self._version_cache:
                version_number = len(self._version_cache[campaign.id]) + 1

            # Create version
            version = CampaignVersion(
                campaign_id=campaign.id,
                version_number=version_number,
                campaign_data=campaign.to_dict(),
                change_description=description,
            )

            # Store in database
            self.db_manager.add_document(
                collection_name="campaigns",
                document_id=f"{campaign.id}_v{version_number}",
                content=json.dumps(version.to_dict()),
                metadata={
                    "type": "campaign_version",
                    "campaign_id": campaign.id,
                    "version_number": version_number,
                    "created_at": version.created_at.isoformat(),
                },
            )

            # Update cache
            if campaign.id not in self._version_cache:
                self._version_cache[campaign.id] = []
            self._version_cache[campaign.id].append(version)

            logger.debug(f"Created version {version_number} for campaign {campaign.id}")
            return True

        except Exception as e:
            logger.error(
                f"Failed to create version for campaign {campaign.id}: {str(e)}",
                extra={
                    "campaign_id": campaign.id,
                    "description": description,
                    "retry_count": retry_count,
                    "error_type": type(e).__name__,
                },
            )

            # Retry mechanism for transient failures
            if retry_count < MAX_RETRIES:
                logger.info(f"Retrying version creation (attempt {retry_count + 1}/{MAX_RETRIES})")
                time.sleep(0.5 * (retry_count + 1))  # Exponential backoff
                return self._create_version(campaign, description, retry_count + 1)

            # Log critical error if all retries failed
            logger.critical(
                f"Failed to create version after {MAX_RETRIES} attempts. "
                f"Campaign data may be lost on rollback. Campaign: {campaign.id}",
                extra={"campaign_id": campaign.id, "description": description},
            )

            # Store failure in a separate collection for manual recovery
            try:
                self.db_manager.add_document(
                    collection_name="campaigns",
                    document_id=f"{campaign.id}_version_failure_{datetime.utcnow().timestamp()}",
                    content=json.dumps(
                        {
                            "campaign_id": campaign.id,
                            "campaign_data": campaign.to_dict(),
                            "description": description,
                            "error": str(e),
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    ),
                    metadata={
                        "type": "version_failure",
                        "campaign_id": campaign.id,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )
            except Exception as recovery_error:
                logger.critical(f"Failed to store version failure record: {str(recovery_error)}")

            return False

    @handle_search_errors()
    async def get_campaign_versions(
        self, campaign_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get version history for a campaign.

        Args:
            campaign_id: Campaign ID
            limit: Maximum number of versions to return

        Returns:
            List of campaign versions
        """
        try:
            docs = self.db_manager.list_documents(
                "campaigns",
                limit=limit,
                metadata_filter={"type": "campaign_version", "campaign_id": campaign_id},
            )

            versions = []
            for doc in docs:
                version_data = json.loads(doc["content"])
                versions.append(
                    {
                        "version_number": version_data["version_number"],
                        "change_description": version_data["change_description"],
                        "created_at": version_data["created_at"],
                    }
                )

            return sorted(versions, key=lambda v: v["version_number"], reverse=True)

        except Exception as e:
            logger.error(f"Failed to get versions: {str(e)}")
            raise DatabaseError(f"Failed to get versions: {str(e)}")

    @handle_search_errors()
    async def rollback_campaign(self, campaign_id: str, version_number: int) -> Dict[str, Any]:
        """
        Rollback campaign to a specific version.

        Args:
            campaign_id: Campaign ID
            version_number: Version to rollback to

        Returns:
            Rollback result
        """
        try:
            # Get the version
            doc = self.db_manager.get_document("campaigns", f"{campaign_id}_v{version_number}")

            if not doc:
                return {
                    "success": False,
                    "message": f"Version {version_number} not found for campaign {campaign_id}",
                }

            # Get version data
            version_data = json.loads(doc["content"])
            campaign_data = version_data["campaign_data"]

            # Create campaign from version
            campaign = Campaign.from_dict(campaign_data)
            campaign.updated_at = datetime.utcnow()

            # Save as current
            self.db_manager.update_document(
                collection_name="campaigns",
                document_id=campaign_id,
                content=json.dumps(campaign.to_dict()),
                metadata={
                    "type": "campaign",
                    "name": campaign.name,
                    "system": campaign.system,
                    "created_at": campaign.created_at.isoformat(),
                    "updated_at": campaign.updated_at.isoformat(),
                },
            )

            # Create new version for rollback
            self._create_version(campaign, f"Rolled back to version {version_number}")

            # Update cache
            self._campaign_cache[campaign_id] = campaign

            return {
                "success": True,
                "message": f"Campaign rolled back to version {version_number}",
                "campaign": campaign.to_dict(),
            }

        except Exception as e:
            logger.error(f"Failed to rollback campaign: {str(e)}")
            raise DatabaseError(f"Failed to rollback campaign: {str(e)}")

    def set_active_campaign(self, campaign_id: str) -> None:
        """
        Set the active campaign for the session.

        Args:
            campaign_id: Campaign ID to set as active
        """
        self.active_campaign_id = campaign_id
        logger.info(f"Active campaign set to: {campaign_id}")

    def get_active_campaign_id(self) -> Optional[str]:
        """
        Get the currently active campaign ID.

        Returns:
            Active campaign ID or None
        """
        return self.active_campaign_id
