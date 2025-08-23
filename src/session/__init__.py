"""
Session management module for TTRPG Assistant MCP Server.
"""

from .mcp_tools import (
    add_monster_to_session,
    add_session_note,
    archive_session,
    complete_session,
    get_session_data,
    initialize_session_tools,
    list_campaign_sessions,
    next_turn,
    register_session_tools,
    set_initiative,
    start_session,
    update_monster_hp,
)
from .models import (
    CombatRound,
    InitiativeEntry,
    Monster,
    MonsterStatus,
    Session,
    SessionNote,
    SessionNoteCategory,
    SessionStatus,
    SessionSummary,
)
from .session_manager import SessionManager

__all__ = [
    # Models
    "Session",
    "SessionStatus",
    "SessionNote",
    "SessionNoteCategory",
    "SessionSummary",
    "Monster",
    "MonsterStatus",
    "InitiativeEntry",
    "CombatRound",
    # Manager
    "SessionManager",
    # MCP Tools
    "initialize_session_tools",
    "register_session_tools",
    "start_session",
    "add_session_note",
    "set_initiative",
    "update_monster_hp",
    "add_monster_to_session",
    "next_turn",
    "complete_session",
    "archive_session",
    "get_session_data",
    "list_campaign_sessions",
]
