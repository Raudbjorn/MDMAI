"""MCP tool implementations for campaign management."""

from typing import Any, Dict, List, Optional

from config.logging_config import get_logger
from src.campaign.campaign_manager import CampaignManager
from src.campaign.rulebook_linker import RulebookLinker
from src.core.database import ChromaDBManager

logger = get_logger(__name__)

# Initialize managers (these will be set by main.py)
db_manager: Optional[ChromaDBManager] = None
campaign_manager: Optional[CampaignManager] = None
rulebook_linker: Optional[RulebookLinker] = None


def initialize_campaign_tools(
    db: ChromaDBManager,
    cm: CampaignManager,
    rl: RulebookLinker
) -> None:
    """
    Initialize campaign tools with required dependencies.
    
    Args:
        db: Database manager
        cm: Campaign manager
        rl: Rulebook linker
    """
    global db_manager, campaign_manager, rulebook_linker
    db_manager = db
    campaign_manager = cm
    rulebook_linker = rl
    logger.info("Campaign MCP tools initialized")


def register_campaign_tools(mcp_server) -> None:
    """
    Register campaign tools with the MCP server.
    
    Args:
        mcp_server: The FastMCP server instance to register tools with
    """
    # Register all campaign management tools
    mcp_server.tool()(create_campaign)
    mcp_server.tool()(get_campaign_data)
    mcp_server.tool()(update_campaign_data)
    mcp_server.tool()(list_campaigns)
    mcp_server.tool()(delete_campaign)
    mcp_server.tool()(get_campaign_versions)
    mcp_server.tool()(rollback_campaign)
    mcp_server.tool()(set_active_campaign)
    mcp_server.tool()(get_campaign_references)
    mcp_server.tool()(validate_campaign_references)
    mcp_server.tool()(search_with_campaign_context)
    logger.info("Campaign MCP tools registered")


async def create_campaign(
    name: str,
    system: str,
    description: str = "",
    setting: str = ""
) -> Dict[str, Any]:
    """
    Create a new campaign.
    
    Args:
        name: Campaign name
        system: Game system (e.g., "D&D 5e", "Pathfinder", "Call of Cthulhu")
        description: Campaign description
        setting: Campaign setting (e.g., "Forgotten Realms", "Homebrew")
    
    Returns:
        Campaign creation result with campaign ID
    """
    if not campaign_manager:
        return {
            "success": False,
            "message": "Campaign manager not initialized"
        }
    
    try:
        result = await campaign_manager.create_campaign(
            name=name,
            system=system,
            description=description,
            setting=setting
        )
        
        # Automatically link to rulebooks
        if result.get("success") and rulebook_linker:
            campaign = await campaign_manager.get_campaign(result["campaign_id"])
            if campaign:
                await rulebook_linker.link_campaign_to_rulebooks(campaign)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to create campaign: {str(e)}")
        return {
            "success": False,
            "message": f"Failed to create campaign: {str(e)}"
        }


async def get_campaign_data(
    campaign_id: str,
    data_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieve campaign-specific data.
    
    Args:
        campaign_id: Campaign ID
        data_type: Type of data to retrieve (characters, npcs, locations, plot_points, or None for all)
    
    Returns:
        Campaign data based on requested type
    """
    if not campaign_manager:
        return {
            "success": False,
            "message": "Campaign manager not initialized"
        }
    
    try:
        return await campaign_manager.get_campaign_data(
            campaign_id=campaign_id,
            data_type=data_type
        )
        
    except Exception as e:
        logger.error(f"Failed to get campaign data: {str(e)}")
        return {
            "success": False,
            "message": f"Failed to get campaign data: {str(e)}"
        }


async def update_campaign_data(
    campaign_id: str,
    data_type: str,
    data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update campaign information.
    
    Args:
        campaign_id: Campaign ID
        data_type: Type of data to update (character, npc, location, plot_point, or campaign)
        data: Data to add or update
    
    Returns:
        Update result
    """
    if not campaign_manager:
        return {
            "success": False,
            "message": "Campaign manager not initialized"
        }
    
    try:
        if data_type == "character":
            result = await campaign_manager.add_character(campaign_id, data)
        elif data_type == "npc":
            result = await campaign_manager.add_npc(campaign_id, data)
        elif data_type == "location":
            result = await campaign_manager.add_location(campaign_id, data)
        elif data_type == "plot_point":
            result = await campaign_manager.add_plot_point(campaign_id, data)
        elif data_type == "campaign":
            result = await campaign_manager.update_campaign(
                campaign_id,
                data,
                data.get("change_description", "Campaign updated")
            )
        else:
            return {
                "success": False,
                "message": f"Unknown data type: {data_type}"
            }
        
        # Re-link rulebooks after update
        if result["success"] and rulebook_linker and data_type != "campaign":
            campaign = await campaign_manager.get_campaign(campaign_id)
            if campaign:
                await rulebook_linker.link_campaign_to_rulebooks(campaign)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to update campaign data: {str(e)}")
        return {
            "success": False,
            "message": f"Failed to update campaign data: {str(e)}"
        }


async def list_campaigns(
    system: Optional[str] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """
    List available campaigns.
    
    Args:
        system: Filter by game system (optional)
        limit: Maximum number of campaigns to return
    
    Returns:
        List of campaign summaries
    """
    if not campaign_manager:
        return {
            "success": False,
            "message": "Campaign manager not initialized"
        }
    
    try:
        campaigns = await campaign_manager.list_campaigns(
            system=system,
            limit=limit
        )
        
        return {
            "success": True,
            "campaigns": campaigns,
            "count": len(campaigns)
        }
        
    except Exception as e:
        logger.error(f"Failed to list campaigns: {str(e)}")
        return {
            "success": False,
            "message": f"Failed to list campaigns: {str(e)}"
        }


async def delete_campaign(campaign_id: str) -> Dict[str, Any]:
    """
    Archive a campaign (soft delete).
    
    Args:
        campaign_id: Campaign ID
    
    Returns:
        Deletion result
    """
    if not campaign_manager:
        return {
            "success": False,
            "message": "Campaign manager not initialized"
        }
    
    try:
        return await campaign_manager.delete_campaign(campaign_id)
        
    except Exception as e:
        logger.error(f"Failed to delete campaign: {str(e)}")
        return {
            "success": False,
            "message": f"Failed to delete campaign: {str(e)}"
        }


async def get_campaign_versions(
    campaign_id: str,
    limit: int = 10
) -> Dict[str, Any]:
    """
    Get version history for a campaign.
    
    Args:
        campaign_id: Campaign ID
        limit: Maximum number of versions to return
    
    Returns:
        List of campaign versions
    """
    if not campaign_manager:
        return {
            "success": False,
            "message": "Campaign manager not initialized"
        }
    
    try:
        versions = await campaign_manager.get_campaign_versions(
            campaign_id=campaign_id,
            limit=limit
        )
        
        return {
            "success": True,
            "versions": versions,
            "count": len(versions)
        }
        
    except Exception as e:
        logger.error(f"Failed to get campaign versions: {str(e)}")
        return {
            "success": False,
            "message": f"Failed to get campaign versions: {str(e)}"
        }


async def rollback_campaign(
    campaign_id: str,
    version_number: int
) -> Dict[str, Any]:
    """
    Rollback a campaign to a previous version.
    
    Args:
        campaign_id: Campaign ID
        version_number: Version number to rollback to
    
    Returns:
        Rollback result
    """
    if not campaign_manager:
        return {
            "success": False,
            "message": "Campaign manager not initialized"
        }
    
    try:
        return await campaign_manager.rollback_campaign(
            campaign_id=campaign_id,
            version_number=version_number
        )
        
    except Exception as e:
        logger.error(f"Failed to rollback campaign: {str(e)}")
        return {
            "success": False,
            "message": f"Failed to rollback campaign: {str(e)}"
        }


async def set_active_campaign(campaign_id: str) -> Dict[str, Any]:
    """
    Set the active campaign for the session.
    
    Args:
        campaign_id: Campaign ID to set as active
    
    Returns:
        Success result
    """
    if not campaign_manager:
        return {
            "success": False,
            "message": "Campaign manager not initialized"
        }
    
    try:
        campaign_manager.set_active_campaign(campaign_id)
        
        return {
            "success": True,
            "message": f"Active campaign set to {campaign_id}",
            "campaign_id": campaign_id
        }
        
    except Exception as e:
        logger.error(f"Failed to set active campaign: {str(e)}")
        return {
            "success": False,
            "message": f"Failed to set active campaign: {str(e)}"
        }


async def get_campaign_references(
    campaign_id: str,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get rulebook references for campaign entities.
    
    Args:
        campaign_id: Campaign ID
        entity_type: Type of entity (character, npc, location, plot_point)
        entity_id: Specific entity ID
    
    Returns:
        Rulebook references
    """
    if not rulebook_linker:
        return {
            "success": False,
            "message": "Rulebook linker not initialized"
        }
    
    try:
        if entity_type and entity_id:
            # Get references for specific entity
            refs = await rulebook_linker.get_references_for_entity(
                campaign_id=campaign_id,
                entity_type=entity_type,
                entity_id=entity_id
            )
            
            return {
                "success": True,
                "references": [ref.to_dict() for ref in refs],
                "count": len(refs)
            }
        else:
            # Get all references for campaign
            campaign = await campaign_manager.get_campaign(campaign_id)
            if not campaign:
                return {
                    "success": False,
                    "message": f"Campaign {campaign_id} not found"
                }
            
            all_refs = await rulebook_linker.link_campaign_to_rulebooks(campaign)
            
            # Format for output
            formatted_refs = {}
            for key, refs in all_refs.items():
                formatted_refs[key] = [ref.to_dict() for ref in refs]
            
            return {
                "success": True,
                "references": formatted_refs,
                "total_count": sum(len(refs) for refs in all_refs.values())
            }
            
    except Exception as e:
        logger.error(f"Failed to get campaign references: {str(e)}")
        return {
            "success": False,
            "message": f"Failed to get campaign references: {str(e)}"
        }


async def validate_campaign_references(campaign_id: str) -> Dict[str, Any]:
    """
    Validate all rulebook references in a campaign.
    
    Args:
        campaign_id: Campaign ID
    
    Returns:
        Validation report with broken references
    """
    if not rulebook_linker:
        return {
            "success": False,
            "message": "Rulebook linker not initialized"
        }
    
    try:
        report = await rulebook_linker.validate_references(campaign_id)
        
        return {
            "success": True,
            "validation_report": report
        }
        
    except Exception as e:
        logger.error(f"Failed to validate references: {str(e)}")
        return {
            "success": False,
            "message": f"Failed to validate references: {str(e)}"
        }


async def search_with_campaign_context(
    query: str,
    campaign_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search rulebooks with campaign context.
    
    Args:
        query: Search query
        campaign_id: Campaign ID for context (uses active campaign if not provided)
    
    Returns:
        Search results with campaign context
    """
    if not campaign_manager or not rulebook_linker:
        return {
            "success": False,
            "message": "Campaign system not initialized"
        }
    
    try:
        # Use active campaign if not specified
        if not campaign_id:
            campaign_id = campaign_manager.get_active_campaign_id()
        
        if not campaign_id:
            return {
                "success": False,
                "message": "No campaign specified or active"
            }
        
        # Get campaign context
        context = await rulebook_linker.get_campaign_context_for_search(
            campaign_id=campaign_id,
            query=query
        )
        
        # Perform search (this would integrate with search service)
        # For now, return the context
        return {
            "success": True,
            "campaign_id": campaign_id,
            "context": context,
            "message": "Search with campaign context ready"
        }
        
    except Exception as e:
        logger.error(f"Failed to search with context: {str(e)}")
        return {
            "success": False,
            "message": f"Failed to search with context: {str(e)}"
        }