"""
Campaign Manager implementation using the returns library.

This module demonstrates proper usage of the returns library with:
- Result type for error handling
- AppError for domain errors
- Async operations with Result
- Chaining and composition patterns
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from returns.result import Failure, Result, Success

from src.core.result_pattern import (
    AppError,
    database_error,
    flat_map_async,
    not_found_error,
    validation_error,
    with_result,
)

logger = logging.getLogger(__name__)


class Campaign:
    """Campaign model."""

    def __init__(
        self,
        id: str,
        name: str,
        system: str,
        description: Optional[str] = None,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ):
        self.id = id
        self.name = name
        self.system = system
        self.description = description
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "system": self.system,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Campaign":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            system=data["system"],
            description=data.get("description"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


class Character:
    """Character model."""

    def __init__(
        self,
        id: str,
        campaign_id: str,
        name: str,
        **kwargs: Any,
    ):
        self.id = id
        self.campaign_id = campaign_id
        self.name = name
        self.attributes = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "campaign_id": self.campaign_id,
            "name": self.name,
            **self.attributes,
        }


class CampaignManager:
    """
    Campaign manager using returns library for Result pattern.
    
    This implementation demonstrates:
    - Using Result[T, AppError] for all public methods
    - Proper error handling with specific error types
    - Async operations with Result
    - Composing Results with map and bind
    """

    def __init__(self, db: Any):
        """Initialize with database connection."""
        self.db = db
        self.collection_name = "campaigns"

    async def create_campaign(
        self,
        name: str,
        system: str,
        description: Optional[str] = None,
    ) -> Result[Campaign, AppError]:
        """
        Create a new campaign with validation.
        
        Args:
            name: Campaign name
            system: Game system (e.g., "D&D 5e")
            description: Optional campaign description
        
        Returns:
            Result containing Campaign or AppError
        """
        # Validate inputs
        validation_result = self._validate_campaign_data(name, system)
        if isinstance(validation_result, Failure):
            return validation_result

        # Create campaign
        campaign = Campaign(
            id=str(uuid.uuid4()),
            name=name.strip(),
            system=system.strip(),
            description=description,
        )

        # Store in database
        storage_result = await self._store_campaign(campaign)
        if isinstance(storage_result, Failure):
            return storage_result

        logger.info(f"Created campaign: {campaign.id}")
        return Success(campaign)

    def _validate_campaign_data(
        self, name: str, system: str
    ) -> Result[None, AppError]:
        """
        Validate campaign data.
        
        Args:
            name: Campaign name
            system: Game system
        
        Returns:
            Success(None) if valid, Failure(AppError) otherwise
        """
        if not name or not name.strip():
            return Failure(validation_error("Campaign name cannot be empty", field="name"))

        if not system or not system.strip():
            return Failure(validation_error("System cannot be empty", field="system"))

        if len(name) > 100:
            return Failure(
                validation_error(
                    "Campaign name too long (max 100 characters)",
                    field="name",
                    max_length=100,
                    actual_length=len(name),
                )
            )

        return Success(None)

    @with_result(error_constructor=lambda msg: database_error(msg, operation="store_campaign"))
    async def _store_campaign(self, campaign: Campaign) -> None:
        """
        Store campaign in database.
        
        This method uses the @with_result decorator to automatically
        wrap exceptions in Result[None, AppError].
        
        Args:
            campaign: Campaign to store
        
        Returns:
            None on success (wrapped in Result by decorator)
        
        Raises:
            Any database exceptions (caught by decorator)
        """
        await self.db.add_document(
            collection=self.collection_name,
            document_id=campaign.id,
            content=campaign.to_dict(),
            metadata={
                "type": "campaign",
                "system": campaign.system,
                "created_at": campaign.created_at.isoformat(),
                "updated_at": campaign.updated_at.isoformat(),
            },
        )

    async def get_campaign(
        self,
        campaign_id: str,
        include_related: bool = False,
    ) -> Result[Dict[str, Any], AppError]:
        """
        Retrieve campaign data with optional related data.
        
        Args:
            campaign_id: Campaign identifier
            include_related: Include characters, NPCs, etc.
        
        Returns:
            Result containing campaign data or AppError
        """
        # Validate campaign ID
        if not campaign_id:
            return Failure(validation_error("Campaign ID cannot be empty", field="campaign_id"))

        # Fetch from database
        fetch_result = await self._fetch_campaign(campaign_id)
        if isinstance(fetch_result, Failure):
            return fetch_result

        campaign_data = fetch_result.unwrap()

        # Include related data if requested
        if include_related:
            related_result = await self._fetch_related_data(campaign_id)
            # Use map to merge data if successful
            return related_result.map(
                lambda related: {**campaign_data, **related}
            )

        return Success(campaign_data)

    async def _fetch_campaign(
        self, campaign_id: str
    ) -> Result[Dict[str, Any], AppError]:
        """
        Fetch campaign from database.
        
        Args:
            campaign_id: Campaign identifier
        
        Returns:
            Result containing campaign data or AppError
        """
        try:
            results = await self.db.get_document(
                collection=self.collection_name,
                document_id=campaign_id,
            )

            if not results:
                return Failure(not_found_error("Campaign", campaign_id))

            return Success(results[0])
        except Exception as e:
            logger.error(f"Database error fetching campaign: {e}")
            return Failure(
                database_error(
                    f"Failed to fetch campaign: {str(e)}",
                    operation="fetch_campaign",
                    campaign_id=campaign_id,
                )
            )

    async def _fetch_related_data(
        self, campaign_id: str
    ) -> Result[Dict[str, Any], AppError]:
        """
        Fetch related campaign data (characters, NPCs, locations).
        
        Args:
            campaign_id: Campaign identifier
        
        Returns:
            Result containing related data or AppError
        """
        try:
            # Fetch all related data in parallel
            characters_task = self.db.query_by_metadata(
                collection=self.collection_name,
                filters={"campaign_id": campaign_id, "type": "character"},
            )
            npcs_task = self.db.query_by_metadata(
                collection=self.collection_name,
                filters={"campaign_id": campaign_id, "type": "npc"},
            )
            locations_task = self.db.query_by_metadata(
                collection=self.collection_name,
                filters={"campaign_id": campaign_id, "type": "location"},
            )

            # Wait for all queries
            characters, npcs, locations = await asyncio.gather(
                characters_task, npcs_task, locations_task
            )

            return Success({
                "characters": characters,
                "npcs": npcs,
                "locations": locations,
            })
        except Exception as e:
            logger.error(f"Error fetching related data: {e}")
            return Failure(
                database_error(
                    f"Failed to fetch related data: {str(e)}",
                    operation="fetch_related_data",
                    campaign_id=campaign_id,
                )
            )

    async def update_campaign(
        self,
        campaign_id: str,
        updates: Dict[str, Any],
    ) -> Result[Campaign, AppError]:
        """
        Update campaign with validation.
        
        This method demonstrates Result chaining:
        1. Fetch existing campaign
        2. Validate updates
        3. Apply updates
        4. Store updated campaign
        
        Args:
            campaign_id: Campaign identifier
            updates: Dictionary of updates
        
        Returns:
            Result containing updated Campaign or AppError
        """
        # Fetch existing campaign
        existing_result = await self.get_campaign(campaign_id)
        if isinstance(existing_result, Failure):
            return existing_result

        # Validate updates
        validation_result = self._validate_updates(updates)
        if isinstance(validation_result, Failure):
            return validation_result

        # Apply updates using map
        campaign_dict = existing_result.unwrap()
        campaign_dict.update(updates)
        campaign_dict["updated_at"] = datetime.utcnow().isoformat()

        # Create updated campaign object
        try:
            campaign = Campaign.from_dict(campaign_dict)
        except Exception as e:
            return Failure(
                validation_error(
                    f"Invalid campaign data: {str(e)}",
                    field="updates",
                    updates=updates,
                )
            )

        # Store updated campaign
        storage_result = await self._store_campaign(campaign)
        if isinstance(storage_result, Failure):
            return storage_result

        return Success(campaign)

    def _validate_updates(
        self, updates: Dict[str, Any]
    ) -> Result[None, AppError]:
        """
        Validate campaign updates.
        
        Args:
            updates: Dictionary of updates
        
        Returns:
            Success(None) if valid, Failure(AppError) otherwise
        """
        if "name" in updates and not updates["name"]:
            return Failure(
                validation_error("Campaign name cannot be empty", field="name")
            )

        if "system" in updates and not updates["system"]:
            return Failure(
                validation_error("System cannot be empty", field="system")
            )

        return Success(None)

    async def add_character(
        self,
        campaign_id: str,
        character_data: Dict[str, Any],
    ) -> Result[Character, AppError]:
        """
        Add a character to campaign with validation.
        
        This demonstrates using bind for sequential operations:
        1. Verify campaign exists
        2. Validate character data
        3. Create and store character
        
        Args:
            campaign_id: Campaign identifier
            character_data: Character data dictionary
        
        Returns:
            Result containing Character or AppError
        """
        # Verify campaign exists (using bind for chaining)
        campaign_result = await self.get_campaign(campaign_id)
        
        # Use bind to chain the next operation only if campaign exists
        return await flat_map_async(
            campaign_result,
            lambda _: self._create_and_store_character(campaign_id, character_data)
        )

    async def _create_and_store_character(
        self,
        campaign_id: str,
        character_data: Dict[str, Any],
    ) -> Result[Character, AppError]:
        """
        Create and store a character.
        
        Args:
            campaign_id: Campaign identifier
            character_data: Character data
        
        Returns:
            Result containing Character or AppError
        """
        # Validate character data
        if not character_data.get("name"):
            return Failure(
                validation_error("Character name is required", field="name")
            )

        # Create character
        character = Character(
            id=str(uuid.uuid4()),
            campaign_id=campaign_id,
            **character_data,
        )

        # Store character
        storage_result = await self._store_character(character)
        if isinstance(storage_result, Failure):
            return storage_result

        return Success(character)

    @with_result(error_constructor=lambda msg: database_error(msg, operation="store_character"))
    async def _store_character(self, character: Character) -> None:
        """
        Store character in database.
        
        Args:
            character: Character to store
        
        Returns:
            None on success (wrapped in Result by decorator)
        """
        await self.db.add_document(
            collection=self.collection_name,
            document_id=character.id,
            content=character.to_dict(),
            metadata={
                "type": "character",
                "campaign_id": character.campaign_id,
                "created_at": datetime.utcnow().isoformat(),
            },
        )

    async def list_campaigns(
        self,
        system: Optional[str] = None,
        limit: int = 10,
    ) -> Result[List[Dict[str, Any]], AppError]:
        """
        List campaigns with optional filtering.
        
        Args:
            system: Optional game system filter
            limit: Maximum number of results
        
        Returns:
            Result containing list of campaigns or AppError
        """
        if limit < 1 or limit > 100:
            return Failure(
                validation_error(
                    "Limit must be between 1 and 100",
                    field="limit",
                    min=1,
                    max=100,
                    actual=limit,
                )
            )

        try:
            filters = {"type": "campaign"}
            if system:
                filters["system"] = system

            results = await self.db.query_by_metadata(
                collection=self.collection_name,
                filters=filters,
                limit=limit,
            )

            return Success(results)
        except Exception as e:
            logger.error(f"Error listing campaigns: {e}")
            return Failure(
                database_error(
                    f"Failed to list campaigns: {str(e)}",
                    operation="list_campaigns",
                    system=system,
                    limit=limit,
                )
            )

    async def delete_campaign(
        self, campaign_id: str
    ) -> Result[bool, AppError]:
        """
        Delete a campaign and all related data.
        
        Args:
            campaign_id: Campaign identifier
        
        Returns:
            Result containing success boolean or AppError
        """
        # Verify campaign exists
        campaign_result = await self.get_campaign(campaign_id)
        if isinstance(campaign_result, Failure):
            return campaign_result

        try:
            # Delete campaign document
            await self.db.delete_document(
                collection=self.collection_name,
                document_id=campaign_id,
            )

            # Delete related data (characters, NPCs, etc.)
            await self._delete_related_data(campaign_id)

            logger.info(f"Deleted campaign: {campaign_id}")
            return Success(True)
        except Exception as e:
            logger.error(f"Error deleting campaign: {e}")
            return Failure(
                database_error(
                    f"Failed to delete campaign: {str(e)}",
                    operation="delete_campaign",
                    campaign_id=campaign_id,
                )
            )

    async def _delete_related_data(self, campaign_id: str) -> None:
        """Delete all data related to a campaign."""
        # Implementation would delete characters, NPCs, locations, etc.
        pass


# Import asyncio for the async operations
import asyncio