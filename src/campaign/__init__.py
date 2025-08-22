"""Campaign management system for TTRPG Assistant."""

from src.campaign.models import (
    Campaign,
    Character,
    NPC,
    Location,
    PlotPoint,
    CampaignVersion,
    CharacterClass
)
from src.campaign.campaign_manager import CampaignManager
from src.campaign.rulebook_linker import RulebookLinker, RulebookReference
from src.campaign.mcp_tools import (
    initialize_campaign_tools,
    register_campaign_tools,
    create_campaign,
    get_campaign_data,
    update_campaign_data,
    list_campaigns,
    delete_campaign,
    get_campaign_versions,
    rollback_campaign,
    set_active_campaign,
    get_campaign_references,
    validate_campaign_references,
    search_with_campaign_context
)

__all__ = [
    # Models
    "Campaign",
    "Character",
    "NPC",
    "Location",
    "PlotPoint",
    "CampaignVersion",
    "CharacterClass",
    
    # Managers
    "CampaignManager",
    "RulebookLinker",
    "RulebookReference",
    
    # MCP Tools
    "initialize_campaign_tools",
    "register_campaign_tools",
    "create_campaign",
    "get_campaign_data",
    "update_campaign_data",
    "list_campaigns",
    "delete_campaign",
    "get_campaign_versions",
    "rollback_campaign",
    "set_active_campaign",
    "get_campaign_references",
    "validate_campaign_references",
    "search_with_campaign_context",
]