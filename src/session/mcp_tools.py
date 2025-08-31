"""
MCP tool definitions for session management.
Implements REQ-008: Session Management
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

from ..search.error_handler import DatabaseError, handle_search_errors
from .models import SessionNoteCategory

logger = logging.getLogger(__name__)

# Global references to be initialized by main.py
_session_manager = None
_campaign_manager = None


def initialize_session_tools(session_manager, campaign_manager):
    """
    Initialize session tools with required dependencies.

    Args:
        session_manager: SessionManager instance
        campaign_manager: CampaignManager instance
    """
    global _session_manager, _campaign_manager
    _session_manager = session_manager
    _campaign_manager = campaign_manager
    logger.info("Session tools initialized")


def register_session_tools(mcp_server):
    """
    Register session management tools with the MCP server.

    Args:
        mcp_server: The FastMCP server instance
    """
    # Register all session tools
    mcp_server.tool()(start_session)
    mcp_server.tool()(add_session_note)
    mcp_server.tool()(set_initiative)
    mcp_server.tool()(update_monster_hp)
    mcp_server.tool()(add_monster_to_session)
    mcp_server.tool()(next_turn)
    mcp_server.tool()(complete_session)
    mcp_server.tool()(get_session_data)
    mcp_server.tool()(list_campaign_sessions)
    mcp_server.tool()(archive_session)

    logger.info("Session tools registered with MCP server")


@handle_search_errors()
async def start_session(campaign_id: str, session_name: str = None) -> Dict[str, Any]:
    """
    Start a new game session for a campaign.

    Args:
        campaign_id: ID of the campaign
        session_name: Optional name for the session

    Returns:
        Session creation result with ID and details
    """
    if not _session_manager:
        raise DatabaseError("Session manager not initialized")

    if not _campaign_manager:
        raise DatabaseError("Campaign manager not initialized")

    # Verify campaign exists
    campaign = await _campaign_manager.get_campaign(campaign_id)
    if not campaign:
        return {"success": False, "error": f"Campaign not found: {campaign_id}"}

    # Create and start the session
    session = await _session_manager.create_session(campaign_id=campaign_id, name=session_name)

    # Start the session immediately
    session = await _session_manager.start_session(session.id)

    return {
        "success": True,
        "message": f"Session '{session.name}' started successfully",
        "id": session.id,
        "data": {
            "session_id": session.id,
            "campaign_id": session.campaign_id,
            "name": session.name,
            "status": session.status.value,
            "date": session.date.isoformat(),
            "current_round": session.current_round,
        },
    }


@handle_search_errors()
async def add_session_note(
    session_id: str, note: str, category: str = "general", tags: List[str] = None
) -> Dict[str, Any]:
    """
    Add a note to the current session.

    Args:
        session_id: ID of the session
        note: Note content
        category: Note category (general, combat, roleplay, loot, quest)
        tags: Optional list of tags

    Returns:
        Note creation result
    """
    if not _session_manager:
        raise DatabaseError("Session manager not initialized")

    # Convert string category to enum
    try:
        category_enum = SessionNoteCategory(category)
    except ValueError:
        valid_categories = [c.value for c in SessionNoteCategory]
        return {
            "success": False,
            "error": f"Invalid category. Must be one of: {', '.join(valid_categories)}",
        }

    note_obj = await _session_manager.add_session_note(
        session_id=session_id, content=note, category=category_enum, tags=tags or []
    )

    return {
        "success": True,
        "message": "Note added to session",
        "data": {
            "note_id": note_obj.id,
            "timestamp": note_obj.timestamp.isoformat(),
            "category": note_obj.category.value,
            "tags": note_obj.tags,
        },
    }


@handle_search_errors()
async def set_initiative(session_id: str, initiative_order: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Set the initiative order for combat.

    Args:
        session_id: ID of the session
        initiative_order: List of initiative entries with format:
            [{"name": str, "initiative": int, "is_player": bool, "is_npc": bool, "is_monster": bool}]

    Returns:
        Initiative setting result
    """
    if not _session_manager:
        raise DatabaseError("Session manager not initialized")

    # Validate initiative entries
    for entry in initiative_order:
        if "name" not in entry or "initiative" not in entry:
            return {
                "success": False,
                "error": "Each initiative entry must have 'name' and 'initiative' fields",
            }

        # Ensure initiative is a number
        try:
            entry["initiative"] = int(entry["initiative"])
        except (ValueError, TypeError):
            return {
                "success": False,
                "error": f"Invalid initiative value for {entry['name']}: must be a number",
            }

    session = await _session_manager.set_initiative(
        session_id=session_id, initiative_order=initiative_order
    )

    # Format the initiative order for response
    formatted_order = [
        {
            "position": idx + 1,
            "name": entry.name,
            "initiative": entry.initiative,
            "type": (
                "player"
                if entry.is_player
                else "npc" if entry.is_npc else "monster" if entry.is_monster else "unknown"
            ),
            "current_turn": entry.current_turn,
        }
        for idx, entry in enumerate(session.initiative_order)
    ]

    return {
        "success": True,
        "message": "Initiative order set",
        "data": {
            "session_id": session_id,
            "initiative_order": formatted_order,
            "current_round": session.current_round,
        },
    }


@handle_search_errors()
async def add_monster_to_session(
    session_id: str,
    name: str,
    monster_type: str,
    max_hp: int,
    armor_class: int = 10,
    challenge_rating: str = "0",
    initiative: int = None,
    notes: str = "",
) -> Dict[str, Any]:
    """
    Add a monster to the session.

    Args:
        session_id: ID of the session
        name: Monster's name
        monster_type: Type of monster (e.g., "Goblin", "Dragon")
        max_hp: Maximum hit points
        armor_class: Armor class (default 10)
        challenge_rating: Challenge rating (default "0")
        initiative: Optional initiative value to add to combat
        notes: Optional notes about the monster

    Returns:
        Monster creation result
    """
    if not _session_manager:
        raise DatabaseError("Session manager not initialized")

    # Create monster data
    monster_data = {
        "name": name,
        "type": monster_type,
        "max_hp": max_hp,
        "current_hp": max_hp,
        "armor_class": armor_class,
        "challenge_rating": challenge_rating,
        "notes": notes,
    }

    monster = await _session_manager.add_monster(
        session_id=session_id, monster_data=monster_data, initiative=initiative
    )

    return {
        "success": True,
        "message": f"Monster '{name}' added to session",
        "data": {
            "monster_id": monster.id,
            "name": monster.name,
            "type": monster.type,
            "hp": f"{monster.current_hp}/{monster.max_hp}",
            "ac": monster.armor_class,
            "cr": monster.challenge_rating,
            "status": monster.status.value,
            "added_to_initiative": initiative is not None,
        },
    }


@handle_search_errors()
async def update_monster_hp(session_id: str, monster_id: str, new_hp: int) -> Dict[str, Any]:
    """
    Update a monster's hit points.

    Args:
        session_id: ID of the session
        monster_id: ID of the monster
        new_hp: New HP value

    Returns:
        Update result with monster status
    """
    if not _session_manager:
        raise DatabaseError("Session manager not initialized")

    monster = await _session_manager.update_monster_hp(
        session_id=session_id, monster_id=monster_id, new_hp=new_hp
    )

    # Determine status message
    status_message = ""
    if monster.status.value == "dead":
        status_message = f"{monster.name} has been defeated!"
    elif monster.status.value == "unconscious":
        status_message = f"{monster.name} is unconscious!"
    elif monster.status.value == "bloodied":
        status_message = f"{monster.name} is bloodied!"

    return {
        "success": True,
        "message": f"Updated {monster.name}'s HP",
        "data": {
            "monster_id": monster.id,
            "name": monster.name,
            "hp": f"{monster.current_hp}/{monster.max_hp}",
            "status": monster.status.value,
            "status_message": status_message,
        },
    }


@handle_search_errors()
async def next_turn(session_id: str) -> Dict[str, Any]:
    """
    Advance to the next turn in initiative order.

    Args:
        session_id: ID of the session

    Returns:
        Information about whose turn it is
    """
    if not _session_manager:
        raise DatabaseError("Session manager not initialized")

    # Get session, next entry, and round completion status in one call
    session, next_entry, round_completed = await _session_manager.next_turn(session_id)

    return {
        "success": True,
        "message": f"It's now {next_entry.name}'s turn",
        "data": {
            "current_turn": {
                "name": next_entry.name,
                "initiative": next_entry.initiative,
                "type": (
                    "player" if next_entry.is_player else "npc" if next_entry.is_npc else "monster"
                ),
            },
            "current_round": session.current_round,
            "round_complete": round_completed,
        },
    }


@handle_search_errors()
async def complete_session(session_id: str) -> Dict[str, Any]:
    """
    Mark a session as completed.

    Args:
        session_id: ID of the session

    Returns:
        Session completion result
    """
    if not _session_manager:
        raise DatabaseError("Session manager not initialized")

    session = await _session_manager.complete_session(session_id)

    # Calculate session statistics
    monsters_defeated = len([m for m in session.monsters if m.status.value == "dead"])
    total_notes = len(session.notes)

    return {
        "success": True,
        "message": f"Session '{session.name}' completed",
        "data": {
            "session_id": session.id,
            "name": session.name,
            "status": session.status.value,
            "completed_at": session.completed_at.isoformat(),
            "statistics": {
                "duration": str(session.completed_at - session.created_at),
                "combat_rounds": session.current_round,
                "monsters_defeated": monsters_defeated,
                "total_notes": total_notes,
            },
        },
    }


@handle_search_errors()
async def archive_session(session_id: str) -> Dict[str, Any]:
    """
    Archive a session for long-term storage.

    Args:
        session_id: ID of the session

    Returns:
        Archive result
    """
    if not _session_manager:
        raise DatabaseError("Session manager not initialized")

    session = await _session_manager.archive_session(session_id)

    return {
        "success": True,
        "message": f"Session '{session.name}' archived",
        "data": {
            "session_id": session.id,
            "name": session.name,
            "status": session.status.value,
            "archived_at": (
                session.completed_at.isoformat()
                if session.completed_at
                else datetime.utcnow().isoformat()
            ),
        },
    }


@handle_search_errors()
async def get_session_data(session_id: str, include_full_details: bool = False) -> Dict[str, Any]:
    """
    Get session data and current state.

    Args:
        session_id: ID of the session
        include_full_details: Whether to include all session details

    Returns:
        Session data
    """
    if not _session_manager:
        raise DatabaseError("Session manager not initialized")

    session = await _session_manager.get_session(session_id)
    if not session:
        return {"success": False, "error": f"Session not found: {session_id}"}

    # Basic session data
    data = {
        "session_id": session.id,
        "campaign_id": session.campaign_id,
        "name": session.name,
        "status": session.status.value,
        "date": session.date.isoformat(),
        "current_round": session.current_round,
        "notes_count": len(session.notes),
        "monsters_count": len(session.monsters),
        "active_monsters": len(session.get_active_monsters()),
    }

    # Add full details if requested
    if include_full_details:
        data["initiative_order"] = [
            {
                "name": entry.name,
                "initiative": entry.initiative,
                "current_turn": entry.current_turn,
                "conditions": entry.conditions,
            }
            for entry in session.initiative_order
        ]

        data["monsters"] = [
            {
                "id": monster.id,
                "name": monster.name,
                "type": monster.type,
                "hp": f"{monster.current_hp}/{monster.max_hp}",
                "status": monster.status.value,
                "conditions": monster.conditions,
            }
            for monster in session.monsters
        ]

        data["recent_notes"] = [
            {
                "timestamp": note.timestamp.isoformat(),
                "category": note.category.value,
                "content": note.content[:100] + "..." if len(note.content) > 100 else note.content,
            }
            for note in sorted(session.notes, key=lambda x: x.timestamp, reverse=True)[:5]
        ]

    return {"success": True, "data": data}


@handle_search_errors()
async def list_campaign_sessions(
    campaign_id: str, include_archived: bool = False
) -> Dict[str, Any]:
    """
    List all sessions for a campaign.

    Args:
        campaign_id: ID of the campaign
        include_archived: Whether to include archived sessions

    Returns:
        List of session summaries
    """
    if not _session_manager:
        raise DatabaseError("Session manager not initialized")

    sessions = await _session_manager.get_campaign_sessions(
        campaign_id=campaign_id, include_archived=include_archived
    )

    session_list = [
        {
            "session_id": session.id,
            "name": session.name,
            "date": session.date.isoformat(),
            "status": session.status.value,
            "notes_count": len(session.notes),
            "monsters_defeated": len([m for m in session.monsters if m.status.value == "dead"]),
            "combat_rounds": session.current_round,
        }
        for session in sessions
    ]

    return {
        "success": True,
        "data": {
            "campaign_id": campaign_id,
            "total_sessions": len(session_list),
            "sessions": session_list,
        },
    }
