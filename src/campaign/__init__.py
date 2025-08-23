"""Campaign management system for TTRPG Assistant."""

from src.campaign.campaign_manager import CampaignManager
from src.campaign.mcp_tools import (
    create_campaign,
    delete_campaign,
    get_campaign_data,
    get_campaign_references,
    get_campaign_versions,
    initialize_campaign_tools,
    list_campaigns,
    register_campaign_tools,
    rollback_campaign,
    search_with_campaign_context,
    set_active_campaign,
    update_campaign_data,
    validate_campaign_references,
)
from src.campaign.models import (
    NPC,
    Campaign,
    CampaignVersion,
    Character,
    CharacterClass,
    Location,
    PlotPoint,
)
from src.campaign.rulebook_linker import RulebookLinker, RulebookReference

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
