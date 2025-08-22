"""
Session management data models for TTRPG Assistant MCP Server.
Implements REQ-008: Session Management
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum
import uuid


class SessionStatus(Enum):
    """Session status enumeration."""
    PLANNED = "planned"
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class MonsterStatus(Enum):
    """Monster status in combat."""
    HEALTHY = "healthy"
    INJURED = "injured"
    BLOODIED = "bloodied"
    UNCONSCIOUS = "unconscious"
    DEAD = "dead"


@dataclass
class InitiativeEntry:
    """Represents an entry in the initiative order."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    initiative: int = 0
    is_player: bool = False
    is_npc: bool = False
    is_monster: bool = False
    entity_id: Optional[str] = None  # Reference to character/npc/monster
    current_turn: bool = False
    conditions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "name": self.name,
            "initiative": self.initiative,
            "is_player": self.is_player,
            "is_npc": self.is_npc,
            "is_monster": self.is_monster,
            "entity_id": self.entity_id,
            "current_turn": self.current_turn,
            "conditions": self.conditions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InitiativeEntry':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Monster:
    """Represents a monster in a session."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: str = ""  # e.g., "Dragon", "Goblin"
    max_hp: int = 0
    current_hp: int = 0
    armor_class: int = 10
    challenge_rating: str = "0"
    status: MonsterStatus = MonsterStatus.HEALTHY
    conditions: List[str] = field(default_factory=list)
    notes: str = ""
    source_reference: Optional[str] = None  # Reference to rulebook entry
    stats: Dict[str, Any] = field(default_factory=dict)  # Full stat block
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "max_hp": self.max_hp,
            "current_hp": self.current_hp,
            "armor_class": self.armor_class,
            "challenge_rating": self.challenge_rating,
            "status": self.status.value,
            "conditions": self.conditions,
            "notes": self.notes,
            "source_reference": self.source_reference,
            "stats": self.stats
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Monster':
        """Create from dictionary."""
        if isinstance(data.get('status'), str):
            data['status'] = MonsterStatus(data['status'])
        return cls(**data)
    
    def update_status(self):
        """Update status based on current HP."""
        hp_percentage = (self.current_hp / self.max_hp) * 100 if self.max_hp > 0 else 0
        
        if self.current_hp <= 0:
            self.status = MonsterStatus.DEAD
        elif self.current_hp <= -self.max_hp:
        if self.current_hp <= -self.max_hp:
            self.status = MonsterStatus.DEAD  # Instant death
        elif self.current_hp <= 0:
            self.status = MonsterStatus.DEAD
        elif self.current_hp == 0:
            self.status = MonsterStatus.UNCONSCIOUS
        elif hp_percentage <= 50:
            self.status = MonsterStatus.BLOODIED
        elif hp_percentage < 100:
            self.status = MonsterStatus.INJURED
        else:
            self.status = MonsterStatus.HEALTHY


@dataclass
class SessionNote:
    """Represents a note in a session."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    content: str = ""
    category: str = "general"  # general, combat, roleplay, loot, quest
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "category": self.category,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionNote':
        """Create from dictionary."""
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class CombatRound:
    """Represents a round of combat."""
    round_number: int = 1
    completed: bool = False
    notes: List[str] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)  # Combat actions taken
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CombatRound':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Session:
    """Represents a game session."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    campaign_id: str = ""
    name: str = ""
    date: datetime = field(default_factory=datetime.utcnow)
    status: SessionStatus = SessionStatus.PLANNED
    notes: List[SessionNote] = field(default_factory=list)
    initiative_order: List[InitiativeEntry] = field(default_factory=list)
    monsters: List[Monster] = field(default_factory=list)
    combat_rounds: List[CombatRound] = field(default_factory=list)
    current_round: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "campaign_id": self.campaign_id,
            "name": self.name,
            "date": self.date.isoformat(),
            "status": self.status.value,
            "notes": [note.to_dict() for note in self.notes],
            "initiative_order": [entry.to_dict() for entry in self.initiative_order],
            "monsters": [monster.to_dict() for monster in self.monsters],
            "combat_rounds": [round.to_dict() for round in self.combat_rounds],
            "current_round": self.current_round,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Session':
        """Create from dictionary."""
        # Handle datetime conversions
        if isinstance(data.get('date'), str):
            data['date'] = datetime.fromisoformat(data['date'])
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data.get('updated_at'), str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if data.get('completed_at') and isinstance(data['completed_at'], str):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        
        # Handle enum conversion
        if isinstance(data.get('status'), str):
            data['status'] = SessionStatus(data['status'])
        
        # Handle nested objects
        if 'notes' in data:
            data['notes'] = [
                SessionNote.from_dict(note) if isinstance(note, dict) else note
                for note in data['notes']
            ]
        
        if 'initiative_order' in data:
            data['initiative_order'] = [
                InitiativeEntry.from_dict(entry) if isinstance(entry, dict) else entry
                for entry in data['initiative_order']
            ]
        
        if 'monsters' in data:
            data['monsters'] = [
                Monster.from_dict(monster) if isinstance(monster, dict) else monster
                for monster in data['monsters']
            ]
        
        if 'combat_rounds' in data:
            data['combat_rounds'] = [
                CombatRound.from_dict(round) if isinstance(round, dict) else round
                for round in data['combat_rounds']
            ]
        
        return cls(**data)
    
    def add_note(self, content: str, category: str = "general", tags: List[str] = None):
        """Add a note to the session."""
        note = SessionNote(
            content=content,
            category=category,
            tags=tags or []
        )
        self.notes.append(note)
        self.updated_at = datetime.utcnow()
        return note
    
    def set_initiative(self, initiative_order: List[Dict[str, Any]]):
        """Set the initiative order for combat."""
        self.initiative_order = []
        for entry_data in initiative_order:
            if isinstance(entry_data, dict):
                entry = InitiativeEntry.from_dict(entry_data)
            else:
                entry = entry_data
            self.initiative_order.append(entry)
        
        # Sort by initiative (descending)
        self.initiative_order.sort(key=lambda x: x.initiative, reverse=True)
        
        # Set first entry as current turn
        if self.initiative_order:
            self.initiative_order[0].current_turn = True
        
        self.updated_at = datetime.utcnow()
    
    def next_turn(self):
        """Advance to the next turn in initiative."""
        if not self.initiative_order:
            return None
        
        # Find current turn
        current_idx = None
        for idx, entry in enumerate(self.initiative_order):
            if entry.current_turn:
                current_idx = idx
                entry.current_turn = False
                break
        
        # Set next turn
        if current_idx is not None:
            next_idx = (current_idx + 1) % len(self.initiative_order)
            self.initiative_order[next_idx].current_turn = True
            
            # Check if we completed a round
            if next_idx == 0:
                self.current_round += 1
                if self.combat_rounds and not self.combat_rounds[-1].completed:
                    self.combat_rounds[-1].completed = True
                self.combat_rounds.append(CombatRound(round_number=self.current_round))
            
            return self.initiative_order[next_idx]
        
        return None
    
    def add_monster(self, monster: Monster, initiative: int = None):
        """Add a monster to the session."""
        self.monsters.append(monster)
        
        # Add to initiative if provided
        if initiative is not None:
            entry = InitiativeEntry(
                name=monster.name,
                initiative=initiative,
                is_monster=True,
                entity_id=monster.id
            )
            self.initiative_order.append(entry)
            self.initiative_order.sort(key=lambda x: x.initiative, reverse=True)
        
        self.updated_at = datetime.utcnow()
        return monster
    
    def update_monster_hp(self, monster_id: str, new_hp: int):
        """Update a monster's HP."""
        for monster in self.monsters:
            if monster.id == monster_id:
                monster.current_hp = new_hp
                monster.update_status()
                self.updated_at = datetime.utcnow()
                return monster
        return None
    
    def get_active_monsters(self) -> List[Monster]:
        """Get all monsters that are not dead."""
        return [m for m in self.monsters if m.status != MonsterStatus.DEAD]
    
    def archive(self):
        """Archive the session."""
        self.status = SessionStatus.ARCHIVED
        if self.status in [SessionStatus.ACTIVE, SessionStatus.PLANNED]:
            self.status = SessionStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()


@dataclass
class SessionSummary:
    """Summary of a session for quick reference."""
    id: str
    campaign_id: str
    name: str
    date: datetime
    status: SessionStatus
    notes_count: int
    monsters_defeated: int
    combat_rounds: int
    
    @classmethod
    def from_session(cls, session: Session) -> 'SessionSummary':
        """Create a summary from a full session."""
        monsters_defeated = len([m for m in session.monsters if m.status == MonsterStatus.DEAD])
        return cls(
            id=session.id,
            campaign_id=session.campaign_id,
            name=session.name,
            date=session.date,
            status=session.status,
            notes_count=len(session.notes),
            monsters_defeated=monsters_defeated,
            combat_rounds=session.current_round
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "campaign_id": self.campaign_id,
            "name": self.name,
            "date": self.date.isoformat(),
            "status": self.status.value,
            "notes_count": self.notes_count,
            "monsters_defeated": self.monsters_defeated,
            "combat_rounds": self.combat_rounds
        }