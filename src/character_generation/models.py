"""Data models for character and NPC generation."""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import uuid


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
    ARTIFICER = "artificer"
    CUSTOM = "custom"


class CharacterRace(Enum):
    """Common character races across TTRPG systems."""
    HUMAN = "human"
    ELF = "elf"
    DWARF = "dwarf"
    HALFLING = "halfling"
    ORC = "orc"
    TIEFLING = "tiefling"
    DRAGONBORN = "dragonborn"
    GNOME = "gnome"
    HALF_ELF = "half-elf"
    HALF_ORC = "half-orc"
    CUSTOM = "custom"


class NPCRole(Enum):
    """Common NPC roles in TTRPGs."""
    MERCHANT = "merchant"
    GUARD = "guard"
    NOBLE = "noble"
    SCHOLAR = "scholar"
    CRIMINAL = "criminal"
    INNKEEPER = "innkeeper"
    PRIEST = "priest"
    ADVENTURER = "adventurer"
    ARTISAN = "artisan"
    COMMONER = "commoner"
    SOLDIER = "soldier"
    MAGE = "mage"
    ASSASSIN = "assassin"
    HEALER = "healer"
    CUSTOM = "custom"


@dataclass
class CharacterStats:
    """Character statistics for TTRPGs."""
    strength: int = 10
    dexterity: int = 10
    constitution: int = 10
    intelligence: int = 10
    wisdom: int = 10
    charisma: int = 10
    
    # Additional stats that vary by system
    hit_points: int = 10
    max_hit_points: int = 10
    armor_class: int = 10
    initiative_bonus: int = 0
    speed: int = 30
    level: int = 1
    experience_points: int = 0
    proficiency_bonus: int = 2
    
    # System-specific stats stored as dict
    custom_stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CharacterStats':
        """Create stats from dictionary."""
        return cls(**data)
    
    def get_modifier(self, stat_value: int) -> int:
        """Calculate D&D-style ability modifier."""
        return (stat_value - 10) // 2


@dataclass
class Equipment:
    """Character equipment and inventory."""
    weapons: List[str] = field(default_factory=list)
    armor: List[str] = field(default_factory=list)
    items: List[str] = field(default_factory=list)
    currency: Dict[str, int] = field(default_factory=lambda: {
        "gold": 0,
        "silver": 0,
        "copper": 0
    })
    magic_items: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert equipment to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Equipment':
        """Create equipment from dictionary."""
        return cls(**data)


@dataclass
class Backstory:
    """Character backstory information."""
    background: str = ""
    personality_traits: List[str] = field(default_factory=list)
    ideals: List[str] = field(default_factory=list)
    bonds: List[str] = field(default_factory=list)
    flaws: List[str] = field(default_factory=list)
    
    # Narrative elements
    origin: str = ""
    motivation: str = ""
    goals: List[str] = field(default_factory=list)
    fears: List[str] = field(default_factory=list)
    relationships: List[Dict[str, str]] = field(default_factory=list)
    
    # System/personality-aware elements
    narrative_style: str = ""  # Matches source personality
    cultural_references: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert backstory to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Backstory':
        """Create backstory from dictionary."""
        return cls(**data)


@dataclass
class PersonalityTrait:
    """NPC personality trait."""
    category: str  # e.g., "demeanor", "motivation", "quirk"
    trait: str
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trait to dictionary."""
        return asdict(self)


@dataclass
class Character:
    """Complete character data model."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    system: str = "D&D 5e"  # Game system
    
    # Core attributes
    character_class: Optional[CharacterClass] = None
    custom_class: Optional[str] = None
    race: Optional[CharacterRace] = None
    custom_race: Optional[str] = None
    alignment: str = "Neutral"
    
    # Character details
    stats: CharacterStats = field(default_factory=CharacterStats)
    equipment: Equipment = field(default_factory=Equipment)
    backstory: Backstory = field(default_factory=Backstory)
    
    # Skills and abilities
    skills: Dict[str, int] = field(default_factory=dict)
    proficiencies: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    spells: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    campaign_id: Optional[str] = None
    player_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert character to dictionary."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.character_class:
            data['character_class'] = self.character_class.value
        if self.race:
            data['race'] = self.race.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Character':
        """Create character from dictionary."""
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data.get('updated_at'), str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        if isinstance(data.get('character_class'), str):
            try:
                data['character_class'] = CharacterClass(data['character_class'])
            except ValueError:
                data['character_class'] = CharacterClass.CUSTOM
                data['custom_class'] = data.get('character_class')
        
        if isinstance(data.get('race'), str):
            try:
                data['race'] = CharacterRace(data['race'])
            except ValueError:
                data['race'] = CharacterRace.CUSTOM
                data['custom_race'] = data.get('race')
        
        if isinstance(data.get('stats'), dict):
            data['stats'] = CharacterStats.from_dict(data['stats'])
        if isinstance(data.get('equipment'), dict):
            data['equipment'] = Equipment.from_dict(data['equipment'])
        if isinstance(data.get('backstory'), dict):
            data['backstory'] = Backstory.from_dict(data['backstory'])
        
        return cls(**data)
    
    def get_class_name(self) -> str:
        """Get the character's class name."""
        if self.character_class == CharacterClass.CUSTOM:
            return self.custom_class or "Unknown"
        return self.character_class.value if self.character_class else "Unknown"
    
    def get_race_name(self) -> str:
        """Get the character's race name."""
        if self.race == CharacterRace.CUSTOM:
            return self.custom_race or "Unknown"
        return self.race.value if self.race else "Unknown"


@dataclass
class NPC(Character):
    """NPC-specific data model extending Character."""
    role: Optional[NPCRole] = None
    custom_role: Optional[str] = None
    
    # NPC-specific attributes
    personality_traits: List[PersonalityTrait] = field(default_factory=list)
    attitude_towards_party: str = "Neutral"  # Friendly, Neutral, Hostile
    importance: str = "Minor"  # Minor, Supporting, Major
    
    # Behavioral attributes
    combat_behavior: str = "Defensive"  # Aggressive, Defensive, Tactical, Flee
    interaction_style: str = "Professional"  # Professional, Friendly, Suspicious, etc.
    knowledge_areas: List[str] = field(default_factory=list)
    secrets: List[str] = field(default_factory=list)
    
    # Location and context
    location: str = ""
    occupation: str = ""
    faction: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert NPC to dictionary."""
        data = super().to_dict()
        if self.role:
            data['role'] = self.role.value
        data['personality_traits'] = [trait.to_dict() for trait in self.personality_traits]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NPC':
        """Create NPC from dictionary."""
        if isinstance(data.get('role'), str):
            try:
                data['role'] = NPCRole(data['role'])
            except ValueError:
                data['role'] = NPCRole.CUSTOM
                data['custom_role'] = data.get('role')
        
        if 'personality_traits' in data:
            traits = []
            for trait_data in data['personality_traits']:
                if isinstance(trait_data, dict):
                    traits.append(PersonalityTrait(**trait_data))
            data['personality_traits'] = traits
        
        # Handle Character base class fields
        character_data = super().from_dict(data)
        return cls(**character_data.__dict__)
    
    def get_role_name(self) -> str:
        """Get the NPC's role name."""
        if self.role == NPCRole.CUSTOM:
            return self.custom_role or "Unknown"
        return self.role.value if self.role else "Unknown"