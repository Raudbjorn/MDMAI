"""Campaign data models for TTRPG Assistant."""

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class CharacterClass(Enum):
    """Common character classes across TTRPG systems."""

    FIGHTER = "fighter"
    WIZARD = "wizard"
    CLERIC = "cleric"
    ROGUE = "rogue"
    RANGER = "ranger"
    PALADIN = "paladin"
    BARBARIAN = "barbarian"
    SORCERER = "sorcerer"
    WARLOCK = "warlock"
    DRUID = "druid"
    MONK = "monk"
    BARD = "bard"
    CUSTOM = "custom"


@dataclass
class Character:
    """Player character data model."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    player_name: str = ""
    character_class: str = ""
    level: int = 1
    race: str = ""
    background: str = ""
    stats: Dict[str, int] = field(default_factory=dict)
    skills: List[str] = field(default_factory=list)
    equipment: List[str] = field(default_factory=list)
    backstory: str = ""
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Character":
        """Create from dictionary representation."""
        data_copy = data.copy()
        if "created_at" in data_copy and isinstance(data_copy["created_at"], str):
            data_copy["created_at"] = datetime.fromisoformat(data_copy["created_at"])
        if "updated_at" in data_copy and isinstance(data_copy["updated_at"], str):
            data_copy["updated_at"] = datetime.fromisoformat(data_copy["updated_at"])
        return cls(**data_copy)


@dataclass
class NPC:
    """Non-player character data model."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    role: str = ""  # merchant, guard, noble, etc.
    location: str = ""
    description: str = ""
    personality_traits: List[str] = field(default_factory=list)
    motivations: str = ""
    stats: Dict[str, int] = field(default_factory=dict)
    relationships: Dict[str, str] = field(default_factory=dict)  # NPC/PC id -> relationship
    quest_hooks: List[str] = field(default_factory=list)
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NPC":
        """Create from dictionary representation."""
        data_copy = data.copy()
        if "created_at" in data_copy and isinstance(data_copy["created_at"], str):
            data_copy["created_at"] = datetime.fromisoformat(data_copy["created_at"])
        if "updated_at" in data_copy and isinstance(data_copy["updated_at"], str):
            data_copy["updated_at"] = datetime.fromisoformat(data_copy["updated_at"])
        return cls(**data_copy)


@dataclass
class Location:
    """Location data model."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    type: str = ""  # city, dungeon, forest, etc.
    description: str = ""
    notable_features: List[str] = field(default_factory=list)
    npcs: List[str] = field(default_factory=list)  # NPC ids
    connected_locations: List[str] = field(default_factory=list)  # Location ids
    secrets: List[str] = field(default_factory=list)
    encounters: List[str] = field(default_factory=list)
    map_reference: str = ""
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Location":
        """Create from dictionary representation."""
        data_copy = data.copy()
        if "created_at" in data_copy and isinstance(data_copy["created_at"], str):
            data_copy["created_at"] = datetime.fromisoformat(data_copy["created_at"])
        if "updated_at" in data_copy and isinstance(data_copy["updated_at"], str):
            data_copy["updated_at"] = datetime.fromisoformat(data_copy["updated_at"])
        return cls(**data_copy)


@dataclass
class PlotPoint:
    """Plot point/quest data model."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    status: str = "planned"  # planned, active, completed, abandoned
    quest_giver: str = ""  # NPC id
    objectives: List[str] = field(default_factory=list)
    rewards: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)  # Location ids
    npcs_involved: List[str] = field(default_factory=list)  # NPC ids
    prerequisites: List[str] = field(default_factory=list)  # PlotPoint ids
    consequences: str = ""
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        if self.completed_at:
            data["completed_at"] = self.completed_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlotPoint":
        """Create from dictionary representation."""
        data_copy = data.copy()
        if "created_at" in data_copy and isinstance(data_copy["created_at"], str):
            data_copy["created_at"] = datetime.fromisoformat(data_copy["created_at"])
        if "updated_at" in data_copy and isinstance(data_copy["updated_at"], str):
            data_copy["updated_at"] = datetime.fromisoformat(data_copy["updated_at"])
        if "completed_at" in data_copy and isinstance(data_copy["completed_at"], str):
            data_copy["completed_at"] = datetime.fromisoformat(data_copy["completed_at"])
        return cls(**data_copy)


@dataclass
class Campaign:
    """Campaign data model."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    system: str = ""  # D&D 5e, Pathfinder, etc.
    description: str = ""
    setting: str = ""
    current_date_ingame: str = ""
    characters: List[Character] = field(default_factory=list)
    npcs: List[NPC] = field(default_factory=list)
    locations: List[Location] = field(default_factory=list)
    plot_points: List[PlotPoint] = field(default_factory=list)
    world_state: Dict[str, Any] = field(default_factory=dict)
    house_rules: List[str] = field(default_factory=list)
    resources: Dict[str, str] = field(default_factory=dict)  # name -> link/reference
    notes: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_session_date: Optional[datetime] = None
    archived: bool = False
    archived_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = {
            "id": self.id,
            "name": self.name,
            "system": self.system,
            "description": self.description,
            "setting": self.setting,
            "current_date_ingame": self.current_date_ingame,
            "characters": [c.to_dict() for c in self.characters],
            "npcs": [n.to_dict() for n in self.npcs],
            "locations": [l.to_dict() for l in self.locations],
            "plot_points": [p.to_dict() for p in self.plot_points],
            "world_state": self.world_state,
            "house_rules": self.house_rules,
            "resources": self.resources,
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
        if self.last_session_date:
            data["last_session_date"] = self.last_session_date.isoformat()
        if self.archived:
            data["archived"] = self.archived
        if self.archived_at:
            data["archived_at"] = self.archived_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Campaign":
        """Create from dictionary representation."""
        # Create a copy to avoid modifying the original
        data_copy = data.copy()

        # Convert datetime strings
        if "created_at" in data_copy and isinstance(data_copy["created_at"], str):
            data_copy["created_at"] = datetime.fromisoformat(data_copy["created_at"])
        if "updated_at" in data_copy and isinstance(data_copy["updated_at"], str):
            data_copy["updated_at"] = datetime.fromisoformat(data_copy["updated_at"])
        if "last_session_date" in data_copy and isinstance(data_copy["last_session_date"], str):
            data_copy["last_session_date"] = datetime.fromisoformat(data_copy["last_session_date"])
        if "archived_at" in data_copy and isinstance(data_copy["archived_at"], str):
            data_copy["archived_at"] = datetime.fromisoformat(data_copy["archived_at"])

        # Convert nested objects
        if "characters" in data_copy:
            data_copy["characters"] = [
                Character.from_dict(c) if isinstance(c, dict) else c
                for c in data_copy["characters"]
            ]
        if "npcs" in data_copy:
            data_copy["npcs"] = [
                NPC.from_dict(n) if isinstance(n, dict) else n for n in data_copy["npcs"]
            ]
        if "locations" in data_copy:
            data_copy["locations"] = [
                Location.from_dict(l) if isinstance(l, dict) else l for l in data_copy["locations"]
            ]
        if "plot_points" in data_copy:
            data_copy["plot_points"] = [
                PlotPoint.from_dict(p) if isinstance(p, dict) else p
                for p in data_copy["plot_points"]
            ]

        return cls(**data_copy)

    def get_character_by_id(self, character_id: str) -> Optional[Character]:
        """Get a character by ID."""
        for character in self.characters:
            if character.id == character_id:
                return character
        return None

    def get_npc_by_id(self, npc_id: str) -> Optional[NPC]:
        """Get an NPC by ID."""
        for npc in self.npcs:
            if npc.id == npc_id:
                return npc
        return None

    def get_location_by_id(self, location_id: str) -> Optional[Location]:
        """Get a location by ID."""
        for location in self.locations:
            if location.id == location_id:
                return location
        return None

    def get_plot_point_by_id(self, plot_id: str) -> Optional[PlotPoint]:
        """Get a plot point by ID."""
        for plot_point in self.plot_points:
            if plot_point.id == plot_id:
                return plot_point
        return None


@dataclass
class CampaignVersion:
    """Campaign version for history tracking."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    campaign_id: str = ""
    version_number: int = 1
    campaign_data: Dict[str, Any] = field(default_factory=dict)
    change_description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CampaignVersion":
        """Create from dictionary representation."""
        data_copy = data.copy()
        if "created_at" in data_copy and isinstance(data_copy["created_at"], str):
            data_copy["created_at"] = datetime.fromisoformat(data_copy["created_at"])
        return cls(**data_copy)
