"""Unit tests for session management system."""

import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, List, Any
import uuid

from src.session.session_manager import SessionManager
from src.session.models import (
    Session, InitiativeEntry, Monster, MonsterStatus,
    SessionNote, CombatRound
)


class TestSessionModels:
    """Test session data models."""
    
    def test_session_creation(self):
        """Test creating a session."""
        session = Session(
            id="session_1",
            campaign_id="campaign_1",
            session_number=1,
            date=datetime.now(),
            title="The Beginning"
        )
        
        assert session.id == "session_1"
        assert session.campaign_id == "campaign_1"
        assert session.session_number == 1
        assert session.title == "The Beginning"
    
    def test_initiative_entry(self):
        """Test initiative tracking."""
        entry = InitiativeEntry(
            id="init_1",
            name="Gandalf",
            initiative=18,
            is_player=True,
            hp_current=50,
            hp_max=50
        )
        
        assert entry.name == "Gandalf"
        assert entry.initiative == 18
        assert entry.is_player is True
        assert entry.hp_current == entry.hp_max
    
    def test_monster_creation(self):
        """Test monster data model."""
        monster = Monster(
            id="monster_1",
            name="Goblin",
            hp_current=7,
            hp_max=7,
            ac=15,
            initiative=12,
            status=MonsterStatus.HEALTHY
        )
        
        assert monster.name == "Goblin"
        assert monster.hp_current == 7
        assert monster.ac == 15
        assert monster.status == MonsterStatus.HEALTHY
    
    def test_monster_status_transitions(self):
        """Test monster status changes."""
        monster = Monster(
            id="monster_1",
            name="Dragon",
            hp_current=200,
            hp_max=200,
            ac=18
        )
        
        # Test status based on HP
        assert monster.status == MonsterStatus.HEALTHY
        
        monster.hp_current = 100  # 50% HP
        monster.update_status()
        assert monster.status == MonsterStatus.INJURED
        
        monster.hp_current = 50  # 25% HP
        monster.update_status()
        assert monster.status == MonsterStatus.BLOODIED
        
        monster.hp_current = 0
        monster.update_status()
        assert monster.status == MonsterStatus.UNCONSCIOUS
        
        monster.hp_current = -10
        monster.update_status()
        assert monster.status == MonsterStatus.DEAD
    
    def test_session_note(self):
        """Test session notes."""
        note = SessionNote(
            id="note_1",
            timestamp=datetime.now(),
            content="The party enters the dungeon",
            tags=["exploration", "dungeon"],
            round_number=1
        )
        
        assert note.content == "The party enters the dungeon"
        assert "exploration" in note.tags
        assert note.round_number == 1
    
    def test_combat_round(self):
        """Test combat round tracking."""
        round_data = CombatRound(
            number=1,
            active_character="Gandalf",
            actions=["Cast Fireball", "Move 30ft"],
            damage_dealt={"Goblin": 28},
            conditions_applied=["Burning"]
        )
        
        assert round_data.number == 1
        assert round_data.active_character == "Gandalf"
        assert "Cast Fireball" in round_data.actions
        assert round_data.damage_dealt["Goblin"] == 28


class TestSessionManager:
    """Test session manager functionality."""
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database."""
        db = Mock()
        db.add_document = AsyncMock()
        db.search = AsyncMock(return_value=[])
        db.update_document = AsyncMock()
        db.delete_document = AsyncMock()
        db.get_document = AsyncMock()
        return db
    
    @pytest.fixture
    def session_manager(self, mock_db):
        """Create session manager with mock database."""
        return SessionManager(mock_db)
    
    @pytest.mark.asyncio
    async def test_create_session(self, session_manager, mock_db):
        """Test creating a new session."""
        session_id = await session_manager.create_session(
            campaign_id="campaign_1",
            title="Session 1: The Beginning",
            session_number=1
        )
        
        assert session_id is not None
        mock_db.add_document.assert_called_once()
        
        # Verify session structure
        call_args = mock_db.add_document.call_args
        assert call_args[1]["collection_name"] == "sessions"
        assert call_args[1]["metadata"]["campaign_id"] == "campaign_1"
    
    @pytest.mark.asyncio
    async def test_get_active_session(self, session_manager, mock_db):
        """Test retrieving active session."""
        mock_db.search.return_value = [{
            "id": "session_1",
            "content": '{"title": "Active Session"}',
            "metadata": {"status": "active"}
        }]
        
        session = await session_manager.get_active_session("campaign_1")
        
        assert session is not None
        assert session["title"] == "Active Session"
        mock_db.search.assert_called()
    
    @pytest.mark.asyncio
    async def test_add_initiative_entry(self, session_manager, mock_db):
        """Test adding to initiative tracker."""
        entry = await session_manager.add_to_initiative(
            session_id="session_1",
            name="Gandalf",
            initiative=18,
            is_player=True,
            hp=50
        )
        
        assert entry is not None
        assert entry["name"] == "Gandalf"
        assert entry["initiative"] == 18
        
        # Should update session
        mock_db.update_document.assert_called()
    
    @pytest.mark.asyncio
    async def test_sort_initiative(self, session_manager):
        """Test initiative sorting."""
        entries = [
            {"name": "Goblin", "initiative": 12},
            {"name": "Gandalf", "initiative": 18},
            {"name": "Frodo", "initiative": 15},
            {"name": "Orc", "initiative": 12}  # Tie
        ]
        
        sorted_entries = session_manager.sort_initiative(entries)
        
        # Should be sorted by initiative (descending)
        assert sorted_entries[0]["name"] == "Gandalf"
        assert sorted_entries[1]["name"] == "Frodo"
        assert sorted_entries[2]["initiative"] == 12  # Tied entries
        assert sorted_entries[3]["initiative"] == 12
    
    @pytest.mark.asyncio
    async def test_add_monster(self, session_manager, mock_db):
        """Test adding a monster to session."""
        monster_id = await session_manager.add_monster(
            session_id="session_1",
            name="Goblin",
            hp=7,
            ac=15,
            initiative=12
        )
        
        assert monster_id is not None
        mock_db.update_document.assert_called()
    
    @pytest.mark.asyncio
    async def test_update_monster_hp(self, session_manager, mock_db):
        """Test updating monster HP."""
        # Mock existing session with monster
        mock_db.get_document.return_value = {
            "id": "session_1",
            "content": '{"monsters": [{"id": "monster_1", "name": "Goblin", "hp_current": 7, "hp_max": 7}]}'
        }
        
        updated = await session_manager.update_monster_hp(
            session_id="session_1",
            monster_id="monster_1",
            damage=3
        )
        
        assert updated is True
        assert updated["hp_current"] == 4
        assert updated["status"] != MonsterStatus.HEALTHY
        
        mock_db.update_document.assert_called()
    
    @pytest.mark.asyncio
    async def test_heal_monster(self, session_manager, mock_db):
        """Test healing a monster."""
        mock_db.get_document.return_value = {
            "id": "session_1",
            "content": '{"monsters": [{"id": "monster_1", "hp_current": 3, "hp_max": 7}]}'
        }
        
        updated = await session_manager.update_monster_hp(
            session_id="session_1",
            monster_id="monster_1",
            damage=-4  # Negative damage = healing
        )
        
        assert updated["hp_current"] == 7  # Can't exceed max HP
    
    @pytest.mark.asyncio
    async def test_add_session_note(self, session_manager, mock_db):
        """Test adding notes to session."""
        note_id = await session_manager.add_note(
            session_id="session_1",
            content="The party finds a secret door",
            tags=["exploration", "discovery"]
        )
        
        assert note_id is not None
        mock_db.update_document.assert_called()
        
        # Verify note structure
        call_args = mock_db.update_document.call_args
        updates = call_args[0][2]  # Third argument is updates
        assert "notes" in updates
    
    @pytest.mark.asyncio
    async def test_combat_round_tracking(self, session_manager, mock_db):
        """Test tracking combat rounds."""
        # Start combat
        await session_manager.start_combat(
            session_id="session_1",
            initial_round=1
        )
        
        # Advance round
        await session_manager.advance_round(
            session_id="session_1",
            actions=["Gandalf casts Fireball"],
            damage_dealt={"Goblin": 28}
        )
        
        mock_db.update_document.assert_called()
        
        # Verify round data
        call_args = mock_db.update_document.call_args
        updates = call_args[0][2]
        assert "current_round" in updates
        assert updates["current_round"] == 2
    
    @pytest.mark.asyncio
    async def test_end_session(self, session_manager, mock_db):
        """Test ending a session."""
        summary = await session_manager.end_session(
            session_id="session_1",
            summary="The party defeated the goblin ambush"
        )
        
        assert summary is not None
        mock_db.update_document.assert_called()
        
        # Verify session marked as completed
        call_args = mock_db.update_document.call_args
        updates = call_args[0][2]
        assert updates["status"] == "completed"
        assert "end_time" in updates
    
    @pytest.mark.asyncio
    async def test_get_session_history(self, session_manager, mock_db):
        """Test retrieving session history."""
        mock_db.search.return_value = [
            {"id": "session_1", "content": '{"session_number": 1}'},
            {"id": "session_2", "content": '{"session_number": 2}'},
            {"id": "session_3", "content": '{"session_number": 3}'}
        ]
        
        history = await session_manager.get_session_history("campaign_1")
        
        assert len(history) == 3
        assert history[0]["session_number"] == 1
        assert history[-1]["session_number"] == 3
    
    @pytest.mark.asyncio
    async def test_session_statistics(self, session_manager, mock_db):
        """Test generating session statistics."""
        mock_db.get_document.return_value = {
            "id": "session_1",
            "content": '''{
                "duration_minutes": 240,
                "combat_rounds": 10,
                "monsters_defeated": 5,
                "notes": [{"content": "note1"}, {"content": "note2"}]
            }'''
        }
        
        stats = await session_manager.get_session_statistics("session_1")
        
        assert stats["duration_hours"] == 4
        assert stats["combat_rounds"] == 10
        assert stats["monsters_defeated"] == 5
        assert stats["notes_count"] == 2


class TestInitiativeTracker:
    """Test initiative tracking functionality."""
    
    @pytest.mark.asyncio
    async def test_initiative_order(self):
        """Test maintaining initiative order."""
        manager = SessionManager(Mock())
        
        # Add entries with various initiatives
        entries = []
        entries.append(manager.create_initiative_entry("PC1", 20, True))
        entries.append(manager.create_initiative_entry("Monster1", 15, False))
        entries.append(manager.create_initiative_entry("PC2", 15, True))  # Tie
        entries.append(manager.create_initiative_entry("Monster2", 10, False))
        
        sorted_entries = manager.sort_initiative(entries)
        
        # Check order
        assert sorted_entries[0]["initiative"] == 20
        assert sorted_entries[-1]["initiative"] == 10
        
        # When tied, order should be stable
        tied_entries = [e for e in sorted_entries if e["initiative"] == 15]
        assert len(tied_entries) == 2
    
    @pytest.mark.asyncio
    async def test_next_turn(self):
        """Test advancing to next turn in initiative."""
        manager = SessionManager(Mock())
        
        entries = [
            {"name": "PC1", "initiative": 20, "id": "1"},
            {"name": "Monster1", "initiative": 15, "id": "2"},
            {"name": "PC2", "initiative": 10, "id": "3"}
        ]
        
        # Start with first
        current = manager.get_next_turn(entries, None)
        assert current["name"] == "PC1"
        
        # Move to second
        current = manager.get_next_turn(entries, "1")
        assert current["name"] == "Monster1"
        
        # Move to third
        current = manager.get_next_turn(entries, "2")
        assert current["name"] == "PC2"
        
        # Wrap around to first
        current = manager.get_next_turn(entries, "3")
        assert current["name"] == "PC1"
    
    @pytest.mark.asyncio
    async def test_remove_from_initiative(self):
        """Test removing entry from initiative."""
        manager = SessionManager(Mock())
        
        entries = [
            {"name": "PC1", "id": "1"},
            {"name": "Monster1", "id": "2"},
            {"name": "PC2", "id": "3"}
        ]
        
        updated = manager.remove_from_initiative(entries, "2")
        
        assert len(updated) == 2
        assert not any(e["id"] == "2" for e in updated)
        assert any(e["id"] == "1" for e in updated)
        assert any(e["id"] == "3" for e in updated)


class TestSessionIntegration:
    """Integration tests for session management."""
    
    @pytest.mark.asyncio
    async def test_complete_combat_workflow(self):
        """Test complete combat workflow."""
        mock_db = Mock()
        mock_db.add_document = AsyncMock()
        mock_db.update_document = AsyncMock()
        mock_db.get_document = AsyncMock()
        
        manager = SessionManager(mock_db)
        
        # Create session
        session_id = await manager.create_session("campaign_1", "Combat Test")
        
        # Add combatants
        await manager.add_to_initiative(session_id, "Gandalf", 18, True, 50)
        await manager.add_to_initiative(session_id, "Frodo", 15, True, 30)
        
        # Add monsters
        goblin1 = await manager.add_monster(session_id, "Goblin 1", 7, 15, 12)
        goblin2 = await manager.add_monster(session_id, "Goblin 2", 7, 15, 14)
        
        # Start combat
        await manager.start_combat(session_id)
        
        # Process round
        await manager.advance_round(session_id, ["Gandalf casts Fireball"])
        
        # Damage monsters
        await manager.update_monster_hp(session_id, goblin1, 7)
        await manager.update_monster_hp(session_id, goblin2, 5)
        
        # Add note
        await manager.add_note(session_id, "Goblins defeated!")
        
        # End session
        await manager.end_session(session_id, "Victory!")
        
        # Verify workflow completed
        assert mock_db.add_document.called
        assert mock_db.update_document.called
    
    @pytest.mark.asyncio
    async def test_session_recovery(self):
        """Test recovering from interrupted session."""
        mock_db = Mock()
        
        # Mock interrupted session data
        mock_db.get_document.return_value = {
            "id": "session_1",
            "content": '''{
                "status": "active",
                "current_round": 5,
                "initiative": [{"name": "PC1", "initiative": 20}],
                "monsters": [{"id": "m1", "hp_current": 3, "hp_max": 7}]
            }'''
        }
        
        manager = SessionManager(mock_db)
        
        # Resume session
        session = await manager.resume_session("session_1")
        
        assert session["status"] == "active"
        assert session["current_round"] == 5
        assert len(session["initiative"]) == 1
        assert len(session["monsters"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])