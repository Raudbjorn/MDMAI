"""
Session management business logic for TTRPG Assistant MCP Server.
Implements REQ-008: Session Management
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from .models import (
    Session, SessionStatus, SessionNote, SessionSummary,
    Monster, MonsterStatus, InitiativeEntry, CombatRound
)
from ..core.database import ChromaDBManager

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages game sessions and related operations."""
    
    def __init__(self, db: ChromaDBManager):
        """Initialize session manager with database connection."""
        self.db = db
        self.collection_name = "sessions"
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure sessions collection exists."""
        try:
            self.db.get_or_create_collection(self.collection_name)
            logger.info(f"Sessions collection ready: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to create sessions collection: {str(e)}")
            raise
    
    async def create_session(
        self,
        campaign_id: str,
        name: str = None,
        date: datetime = None
    ) -> Session:
        """
        Create a new game session.
        
        Args:
            campaign_id: ID of the campaign this session belongs to
            name: Optional session name
            date: Optional session date (defaults to now)
        
        Returns:
            Created Session object
        """
        try:
            # Generate session name if not provided
            if not name:
                # Count existing sessions for this campaign
                existing_sessions = await self.get_campaign_sessions(campaign_id)
                session_number = len(existing_sessions) + 1
                name = f"Session {session_number}"
            
            # Create session
            session = Session(
                campaign_id=campaign_id,
                name=name,
                date=date or datetime.utcnow(),
                status=SessionStatus.PLANNED
            )
            
            # Store in database
            await self._store_session(session)
            
            logger.info(f"Created session: {session.id} for campaign: {campaign_id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to create session: {str(e)}")
            raise
    
    async def start_session(self, session_id: str) -> Session:
        """
        Start a game session (change status to ACTIVE).
        
        Args:
            session_id: ID of the session to start
        
        Returns:
            Updated Session object
        """
        try:
            session = await self.get_session(session_id)
            if not session:
                raise ValueError(f"Session not found: {session_id}")
            
            if session.status == SessionStatus.COMPLETED:
                raise ValueError("Cannot start a completed session")
            
            if session.status == SessionStatus.ARCHIVED:
                raise ValueError("Cannot start an archived session")
            
            session.status = SessionStatus.ACTIVE
            session.updated_at = datetime.utcnow()
            
            # Initialize first combat round if needed
            if not session.combat_rounds:
                session.combat_rounds.append(CombatRound(round_number=1))
                session.current_round = 1
            else:
                # Ensure current_round is consistent with the latest round
                session.current_round = max(
                    (cr.round_number for cr in session.combat_rounds),
                    default=1
                )
            await self._store_session(session)
            
            logger.info(f"Started session: {session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to start session: {str(e)}")
            raise
    
    async def add_session_note(
        self,
        session_id: str,
        content: str,
        category: str = "general",
        tags: List[str] = None
    ) -> SessionNote:
        """
        Add a note to a session.
        
        Args:
            session_id: ID of the session
            content: Note content
            category: Note category (general, combat, roleplay, loot, quest)
            tags: Optional list of tags
        
        Returns:
            Created SessionNote object
        """
        try:
            session = await self.get_session(session_id)
            if not session:
                raise ValueError(f"Session not found: {session_id}")
            
            note = session.add_note(content, category, tags)
            await self._store_session(session)
            
            logger.info(f"Added note to session: {session_id}")
            return note
            
        except Exception as e:
            logger.error(f"Failed to add session note: {str(e)}")
            raise
    
    async def set_initiative(
        self,
        session_id: str,
        initiative_order: List[Dict[str, Any]]
    ) -> Session:
        """
        Set the initiative order for a session.
        
        Args:
            session_id: ID of the session
            initiative_order: List of initiative entries
        
        Returns:
            Updated Session object
        """
        try:
            session = await self.get_session(session_id)
            if not session:
                raise ValueError(f"Session not found: {session_id}")
            
            session.set_initiative(initiative_order)
            await self._store_session(session)
            
            logger.info(f"Set initiative for session: {session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to set initiative: {str(e)}")
            raise
    
    async def add_monster(
        self,
        session_id: str,
        monster_data: Dict[str, Any],
        initiative: int = None
    ) -> Monster:
        """
        Add a monster to a session.
        
        Args:
            session_id: ID of the session
            monster_data: Monster information
            initiative: Optional initiative value
        
        Returns:
            Created Monster object
        """
        try:
            session = await self.get_session(session_id)
            if not session:
                raise ValueError(f"Session not found: {session_id}")
            
            # Create monster from data
            monster = Monster.from_dict(monster_data) if isinstance(monster_data, dict) else monster_data
            
            # Set current HP to max if not specified
            if monster.current_hp == 0 and monster.max_hp > 0:
                monster.current_hp = monster.max_hp
            
            session.add_monster(monster, initiative)
            await self._store_session(session)
            
            logger.info(f"Added monster to session: {session_id}")
            return monster
            
        except Exception as e:
            logger.error(f"Failed to add monster: {str(e)}")
            raise
    
    async def update_monster_hp(
        self,
        session_id: str,
        monster_id: str,
        new_hp: int
    ) -> Monster:
        """
        Update a monster's hit points.
        
        Args:
            session_id: ID of the session
            monster_id: ID of the monster
            new_hp: New HP value
        
        Returns:
            Updated Monster object
        """
        try:
            session = await self.get_session(session_id)
            if not session:
                raise ValueError(f"Session not found: {session_id}")
            
            monster = session.update_monster_hp(monster_id, new_hp)
            if not monster:
                raise ValueError(f"Monster not found: {monster_id}")
            
            await self._store_session(session)
            
            logger.info(f"Updated monster HP in session: {session_id}")
            return monster
            
        except Exception as e:
            logger.error(f"Failed to update monster HP: {str(e)}")
            raise
    
    async def next_turn(self, session_id: str) -> InitiativeEntry:
        """
        Advance to the next turn in initiative.
        
        Args:
            session_id: ID of the session
        
        Returns:
            The entry whose turn it is now
        """
        try:
            session = await self.get_session(session_id)
            if not session:
                raise ValueError(f"Session not found: {session_id}")
            
            next_entry = session.next_turn()
            if not next_entry:
                raise ValueError("No initiative order set")
            
            await self._store_session(session)
            
            logger.info(f"Advanced turn in session: {session_id}")
            return next_entry
            
        except Exception as e:
            logger.error(f"Failed to advance turn: {str(e)}")
            raise
    
    async def complete_session(self, session_id: str) -> Session:
        """
        Mark a session as completed.
        
        Args:
            session_id: ID of the session
        
        Returns:
            Updated Session object
        """
        try:
            session = await self.get_session(session_id)
            if not session:
                raise ValueError(f"Session not found: {session_id}")
            
            session.status = SessionStatus.COMPLETED
            session.completed_at = datetime.utcnow()
            session.updated_at = datetime.utcnow()
            
            await self._store_session(session)
            
            logger.info(f"Completed session: {session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to complete session: {str(e)}")
            raise
    
    async def archive_session(self, session_id: str) -> Session:
        """
        Archive a session.
        
        Args:
            session_id: ID of the session
        
        Returns:
            Updated Session object
        """
        try:
            session = await self.get_session(session_id)
            if not session:
                raise ValueError(f"Session not found: {session_id}")
            
            session.archive()
            await self._store_session(session)
            
            logger.info(f"Archived session: {session_id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to archive session: {str(e)}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session by ID.
        
        Args:
            session_id: ID of the session
        
        Returns:
            Session object or None if not found
        """
        try:
            collection = self.db.get_or_create_collection(self.collection_name)
            results = collection.get(
                ids=[session_id],
                include=["documents", "metadatas"]
            )
            
            if results and results['documents']:
                session_data = json.loads(results['documents'][0])
                return Session.from_dict(session_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session: {str(e)}")
            return None
    
    async def get_campaign_sessions(
        self,
        campaign_id: str,
        include_archived: bool = False
    ) -> List[Session]:
        """
        Get all sessions for a campaign.
        
        Args:
            campaign_id: ID of the campaign
            include_archived: Whether to include archived sessions
        
        Returns:
            List of Session objects
        """
        try:
            collection = self.db.get_or_create_collection(self.collection_name)
            
            # Build filter
            where_filter = {"campaign_id": campaign_id}
            if not include_archived:
                where_filter["status"] = {"$ne": SessionStatus.ARCHIVED.value}
            
            results = collection.get(
                where=where_filter,
                include=["documents", "metadatas"]
            )
            
            sessions = []
            if results and results['documents']:
                for doc in results['documents']:
                    session_data = json.loads(doc)
                    sessions.append(Session.from_dict(session_data))
            
            # Sort by date
            sessions.sort(key=lambda x: x.date, reverse=True)
            
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to get campaign sessions: {str(e)}")
            return []
    
    async def get_active_sessions(self) -> List[Session]:
        """
        Get all currently active sessions.
        
        Returns:
            List of active Session objects
        """
        try:
            collection = self.db.get_or_create_collection(self.collection_name)
            
            results = collection.get(
                where={"status": SessionStatus.ACTIVE.value},
                include=["documents", "metadatas"]
            )
            
            sessions = []
            if results and results['documents']:
                for doc in results['documents']:
                    session_data = json.loads(doc)
                    sessions.append(Session.from_dict(session_data))
            
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to get active sessions: {str(e)}")
            return []
    
    async def get_session_summary(self, session_id: str) -> Optional[SessionSummary]:
        """
        Get a summary of a session.
        
        Args:
            session_id: ID of the session
        
        Returns:
            SessionSummary object or None if not found
        """
        try:
            session = await self.get_session(session_id)
            if session:
                return SessionSummary.from_session(session)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session summary: {str(e)}")
            return None
    
    async def search_session_notes(
        self,
        session_id: str,
        query: str = None,
        category: str = None,
        tags: List[str] = None
    ) -> List[SessionNote]:
        """
        Search notes within a session.
        
        Args:
            session_id: ID of the session
            query: Optional text query
            category: Optional category filter
            tags: Optional tag filters
        
        Returns:
            List of matching SessionNote objects
        """
        try:
            session = await self.get_session(session_id)
            if not session:
                return []
            
            notes = session.notes
            
            # Filter by category
            if category:
                notes = [n for n in notes if n.category == category]
            
            # Filter by tags
            if tags:
                notes = [n for n in notes if any(tag in n.tags for tag in tags)]
            
            # Filter by query
            if query:
                query_lower = query.lower()
                notes = [n for n in notes if query_lower in n.content.lower()]
            
            # Sort by timestamp (most recent first)
            notes.sort(key=lambda x: x.timestamp, reverse=True)
            
            return notes
            
        except Exception as e:
            logger.error(f"Failed to search session notes: {str(e)}")
            return []
    
    async def _store_session(self, session: Session):
        """Store a session in the database."""
        try:
            collection = self.db.get_or_create_collection(self.collection_name)
            
            # Prepare metadata
            metadata = {
                "campaign_id": session.campaign_id,
                "status": session.status.value,
                "date": session.date.isoformat(),
                "updated_at": session.updated_at.isoformat()
            }
            
            # Store session
            collection.upsert(
                ids=[session.id],
                documents=[json.dumps(session.to_dict())],
                metadatas=[metadata]
            )
            
        except Exception as e:
            logger.error(f"Failed to store session: {str(e)}")
            raise
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session (for testing or cleanup).
        
        Args:
            session_id: ID of the session to delete
        
        Returns:
            True if deleted, False otherwise
        """
        try:
            collection = self.db.get_or_create_collection(self.collection_name)
            collection.delete(ids=[session_id])
            logger.info(f"Deleted session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete session: {str(e)}")
            return False