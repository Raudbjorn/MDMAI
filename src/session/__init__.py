"""
Session management module for TTRPG Assistant MCP Server.
"""

from .models import (
    Session,
    SessionStatus,
    SessionNote,
    SessionNoteCategory,
    SessionSummary,
    Monster,
    MonsterStatus,
    InitiativeEntry,
    CombatRound
)

from .session_manager import SessionManager

from .mcp_tools import (
    initialize_session_tools,
    register_session_tools,
    start_session,
    add_session_note,
    set_initiative,
    update_monster_hp,
    add_monster_to_session,
    next_turn,
    complete_session,
    archive_session,
    get_session_data,
    list_campaign_sessions
)

__all__ = [
    # Models
    'Session',
    'SessionStatus',
    'SessionNote',
    'SessionNoteCategory',
    'SessionSummary',
    'Monster',
    'MonsterStatus',
    'InitiativeEntry',
    'CombatRound',
    
    # Manager
    'SessionManager',
    
    # MCP Tools
    'initialize_session_tools',
    'register_session_tools',
    'start_session',
    'add_session_note',
    'set_initiative',
    'update_monster_hp',
    'add_monster_to_session',
    'next_turn',
    'complete_session',
    'archive_session',
    'get_session_data',
    'list_campaign_sessions'
]