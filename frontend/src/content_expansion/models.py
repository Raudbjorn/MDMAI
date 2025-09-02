"""
Extended data models for multi-genre TTRPG content extraction.

This module defines comprehensive data structures for storing and classifying
content extracted from various TTRPG rulebooks across different genres.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Any
from pathlib import Path

# Import TTRPGGenre from the main models to avoid duplication
try:
    from src.character_generation.models import TTRPGGenre
except ImportError:
    # Fallback definition if import fails
    class TTRPGGenre(Enum):
        """Enumeration of TTRPG genres for classification."""
        FANTASY = "fantasy"
        SCI_FI = "sci_fi"
        CYBERPUNK = "cyberpunk"
        COSMIC_HORROR = "cosmic_horror"
        POST_APOCALYPTIC = "post_apocalyptic"
        STEAMPUNK = "steampunk"
        URBAN_FANTASY = "urban_fantasy"
        SPACE_OPERA = "space_opera"
        SUPERHERO = "superhero"
        HISTORICAL = "historical"
        WESTERN = "western"
        NOIR = "noir"
        PULP = "pulp"
        MODERN = "modern"
        MILITARY = "military"
        HORROR = "horror"
        MYSTERY = "mystery"
        MYTHOLOGICAL = "mythological"
        ANIME = "anime"
        GENERIC = "generic"
        UNKNOWN = "unknown"


class ContentType(Enum):
    """Types of content that can be extracted from PDFs."""
    RACE = auto()
    CLASS = auto()
    NPC = auto()
    EQUIPMENT = auto()
    SPELL = auto()
    FEAT = auto()
    SKILL = auto()
    MONSTER = auto()
    LOCATION = auto()
    RULE = auto()
    LORE = auto()
    ADVENTURE = auto()
    ITEM = auto()
    VEHICLE = auto()
    FACTION = auto()
    DEITY = auto()
    BACKGROUND = auto()
    TRAIT = auto()
    CONDITION = auto()
    OTHER = auto()


class ExtractionConfidence(Enum):
    """Confidence levels for extracted content."""
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    UNCERTAIN = auto()


@dataclass
class SourceAttribution:
    """Track the source of extracted content."""
    pdf_path: Path
    pdf_name: str
    page_number: int
    section_title: Optional[str] = None
    extraction_timestamp: datetime = field(default_factory=datetime.utcnow)
    confidence: ExtractionConfidence = ExtractionConfidence.MEDIUM
    extraction_method: str = "pattern_matching"
    notes: Optional[str] = None


@dataclass
class ExtendedCharacterRace:
    """Extended character race supporting multiple genre types."""
    name: str
    genre: TTRPGGenre
    description: str
    traits: List[str] = field(default_factory=list)
    abilities: Dict[str, Any] = field(default_factory=dict)
    stat_modifiers: Dict[str, int] = field(default_factory=dict)
    size: str = "Medium"
    speed: str = "30 ft"
    languages: List[str] = field(default_factory=list)
    subraces: List[str] = field(default_factory=list)
    special_features: List[str] = field(default_factory=list)
    restrictions: List[str] = field(default_factory=list)
    source: Optional[SourceAttribution] = None
    tags: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "genre": self.genre.name,
            "description": self.description,
            "traits": self.traits,
            "abilities": self.abilities,
            "stat_modifiers": self.stat_modifiers,
            "size": self.size,
            "speed": self.speed,
            "languages": self.languages,
            "subraces": self.subraces,
            "special_features": self.special_features,
            "restrictions": self.restrictions,
            "source": {
                "pdf_name": self.source.pdf_name,
                "page_number": self.source.page_number,
                "section_title": self.source.section_title,
                "confidence": self.source.confidence.name,
            } if self.source else None,
            "tags": list(self.tags),
        }


@dataclass
class ExtendedCharacterClass:
    """Extended character class supporting multiple genre types."""
    name: str
    genre: TTRPGGenre
    description: str
    hit_dice: Optional[str] = None
    primary_ability: Optional[str] = None
    saves: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    equipment: List[str] = field(default_factory=list)
    features: Dict[int, List[str]] = field(default_factory=dict)  # Level -> Features
    subclasses: List[str] = field(default_factory=list)
    spell_casting: Optional[Dict[str, Any]] = None
    prerequisites: List[str] = field(default_factory=list)
    progression_table: Optional[Dict[int, Dict[str, Any]]] = None
    source: Optional[SourceAttribution] = None
    tags: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "genre": self.genre.name,
            "description": self.description,
            "hit_dice": self.hit_dice,
            "primary_ability": self.primary_ability,
            "saves": self.saves,
            "skills": self.skills,
            "equipment": self.equipment,
            "features": {str(k): v for k, v in self.features.items()},
            "subclasses": self.subclasses,
            "spell_casting": self.spell_casting,
            "prerequisites": self.prerequisites,
            "progression_table": {str(k): v for k, v in self.progression_table.items()} if self.progression_table else None,
            "source": {
                "pdf_name": self.source.pdf_name,
                "page_number": self.source.page_number,
                "section_title": self.source.section_title,
                "confidence": self.source.confidence.name,
            } if self.source else None,
            "tags": list(self.tags),
        }


@dataclass
class ExtendedNPCRole:
    """Extended NPC role supporting multiple genre types."""
    name: str
    genre: TTRPGGenre
    description: str
    role_type: str  # e.g., "Combat", "Social", "Support", "Boss"
    challenge_rating: Optional[str] = None
    abilities: Dict[str, Any] = field(default_factory=dict)
    stats: Dict[str, int] = field(default_factory=dict)
    skills: List[str] = field(default_factory=list)
    traits: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    reactions: List[str] = field(default_factory=list)
    legendary_actions: List[str] = field(default_factory=list)
    equipment: List[str] = field(default_factory=list)
    motivation: Optional[str] = None
    tactics: Optional[str] = None
    loot: List[str] = field(default_factory=list)
    source: Optional[SourceAttribution] = None
    tags: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "genre": self.genre.name,
            "description": self.description,
            "role_type": self.role_type,
            "challenge_rating": self.challenge_rating,
            "abilities": self.abilities,
            "stats": self.stats,
            "skills": self.skills,
            "traits": self.traits,
            "actions": self.actions,
            "reactions": self.reactions,
            "legendary_actions": self.legendary_actions,
            "equipment": self.equipment,
            "motivation": self.motivation,
            "tactics": self.tactics,
            "loot": self.loot,
            "source": {
                "pdf_name": self.source.pdf_name,
                "page_number": self.source.page_number,
                "section_title": self.source.section_title,
                "confidence": self.source.confidence.name,
            } if self.source else None,
            "tags": list(self.tags),
        }


@dataclass
class ExtendedEquipment:
    """Extended equipment supporting multiple genre types."""
    name: str
    genre: TTRPGGenre
    equipment_type: str  # weapon, armor, tool, consumable, etc.
    description: str
    cost: Optional[str] = None
    weight: Optional[str] = None
    properties: List[str] = field(default_factory=list)
    damage: Optional[str] = None
    armor_class: Optional[str] = None
    requirements: List[str] = field(default_factory=list)
    special_abilities: List[str] = field(default_factory=list)
    tech_level: Optional[str] = None  # For sci-fi/cyberpunk
    rarity: Optional[str] = None
    attunement: bool = False
    source: Optional[SourceAttribution] = None
    tags: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "genre": self.genre.name,
            "equipment_type": self.equipment_type,
            "description": self.description,
            "cost": self.cost,
            "weight": self.weight,
            "properties": self.properties,
            "damage": self.damage,
            "armor_class": self.armor_class,
            "requirements": self.requirements,
            "special_abilities": self.special_abilities,
            "tech_level": self.tech_level,
            "rarity": self.rarity,
            "attunement": self.attunement,
            "source": {
                "pdf_name": self.source.pdf_name,
                "page_number": self.source.page_number,
                "section_title": self.source.section_title,
                "confidence": self.source.confidence.name,
            } if self.source else None,
            "tags": list(self.tags),
        }


@dataclass
class ExtractedContent:
    """Container for all extracted content from a PDF."""
    pdf_path: Path
    pdf_name: str
    genre: TTRPGGenre
    races: List[ExtendedCharacterRace] = field(default_factory=list)
    classes: List[ExtendedCharacterClass] = field(default_factory=list)
    npcs: List[ExtendedNPCRole] = field(default_factory=list)
    equipment: List[ExtendedEquipment] = field(default_factory=list)
    other_content: Dict[ContentType, List[Dict[str, Any]]] = field(default_factory=dict)
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pdf_path": str(self.pdf_path),
            "pdf_name": self.pdf_name,
            "genre": self.genre.name,
            "races": [r.to_dict() for r in self.races],
            "classes": [c.to_dict() for c in self.classes],
            "npcs": [n.to_dict() for n in self.npcs],
            "equipment": [e.to_dict() for e in self.equipment],
            "other_content": {
                k.name: v for k, v in self.other_content.items()
            },
            "extraction_metadata": self.extraction_metadata,
            "errors": self.errors,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractedContent":
        """Create ExtractedContent from dictionary (for safe deserialization)."""
        from pathlib import Path
        
        # Convert genre string back to enum
        genre = TTRPGGenre[data["genre"]] if isinstance(data["genre"], str) else data["genre"]
        
        # Create instance
        content = cls(
            pdf_path=Path(data["pdf_path"]),
            pdf_name=data["pdf_name"],
            genre=genre
        )
        
        # Reconstruct races
        for race_data in data.get("races", []):
            race = ExtendedCharacterRace(
                name=race_data["name"],
                genre=TTRPGGenre[race_data["genre"]],
                description=race_data["description"],
                traits=race_data.get("traits", []),
                abilities=race_data.get("abilities", {}),
                stat_modifiers=race_data.get("stat_modifiers", {}),
                size=race_data.get("size", "Medium"),
                speed=race_data.get("speed", "30 ft"),
                languages=race_data.get("languages", []),
                subraces=race_data.get("subraces", []),
                special_features=race_data.get("special_features", []),
                restrictions=race_data.get("restrictions", []),
                tags=set(race_data.get("tags", []))
            )
            content.races.append(race)
        
        # Reconstruct classes
        for class_data in data.get("classes", []):
            char_class = ExtendedCharacterClass(
                name=class_data["name"],
                genre=TTRPGGenre[class_data["genre"]],
                description=class_data["description"],
                hit_dice=class_data.get("hit_dice"),
                primary_ability=class_data.get("primary_ability"),
                saves=class_data.get("saves", []),
                skills=class_data.get("skills", []),
                equipment=class_data.get("equipment", []),
                features={int(k): v for k, v in class_data.get("features", {}).items()},
                subclasses=class_data.get("subclasses", []),
                spell_casting=class_data.get("spell_casting"),
                prerequisites=class_data.get("prerequisites", []),
                progression_table={int(k): v for k, v in class_data["progression_table"].items()} 
                    if class_data.get("progression_table") else None,
                tags=set(class_data.get("tags", []))
            )
            content.classes.append(char_class)
        
        # Reconstruct NPCs
        for npc_data in data.get("npcs", []):
            npc = ExtendedNPCRole(
                name=npc_data["name"],
                genre=TTRPGGenre[npc_data["genre"]],
                description=npc_data["description"],
                role_type=npc_data["role_type"],
                challenge_rating=npc_data.get("challenge_rating"),
                abilities=npc_data.get("abilities", {}),
                stats=npc_data.get("stats", {}),
                skills=npc_data.get("skills", []),
                traits=npc_data.get("traits", []),
                actions=npc_data.get("actions", []),
                reactions=npc_data.get("reactions", []),
                legendary_actions=npc_data.get("legendary_actions", []),
                equipment=npc_data.get("equipment", []),
                motivation=npc_data.get("motivation"),
                tactics=npc_data.get("tactics"),
                loot=npc_data.get("loot", []),
                tags=set(npc_data.get("tags", []))
            )
            content.npcs.append(npc)
        
        # Reconstruct equipment
        for equip_data in data.get("equipment", []):
            equipment = ExtendedEquipment(
                name=equip_data["name"],
                genre=TTRPGGenre[equip_data["genre"]],
                equipment_type=equip_data["equipment_type"],
                description=equip_data["description"],
                cost=equip_data.get("cost"),
                weight=equip_data.get("weight"),
                properties=equip_data.get("properties", []),
                damage=equip_data.get("damage"),
                armor_class=equip_data.get("armor_class"),
                requirements=equip_data.get("requirements", []),
                special_abilities=equip_data.get("special_abilities", []),
                tech_level=equip_data.get("tech_level"),
                rarity=equip_data.get("rarity"),
                attunement=equip_data.get("attunement", False),
                tags=set(equip_data.get("tags", []))
            )
            content.equipment.append(equipment)
        
        # Reconstruct other content
        other_content_dict = data.get("other_content", {})
        for content_type_str, items in other_content_dict.items():
            try:
                content_type = ContentType[content_type_str.upper()]
                content.other_content[content_type] = items
            except KeyError:
                # Skip unknown content types
                pass
        
        # Copy metadata and errors
        content.extraction_metadata = data.get("extraction_metadata", {})
        content.errors = data.get("errors", [])
        
        return content
    
    def get_summary(self) -> Dict[str, int]:
        """Get a summary of extracted content counts."""
        summary = {
            "races": len(self.races),
            "classes": len(self.classes),
            "npcs": len(self.npcs),
            "equipment": len(self.equipment),
        }
        for content_type, items in self.other_content.items():
            summary[content_type.name.lower()] = len(items)
        return summary


@dataclass
class ProcessingState:
    """Track the state of PDF processing for resumable operations."""
    total_pdfs: int
    processed_pdfs: int = 0
    failed_pdfs: List[str] = field(default_factory=list)
    successful_pdfs: List[str] = field(default_factory=list)
    current_pdf: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.utcnow)
    last_checkpoint: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_pdfs": self.total_pdfs,
            "processed_pdfs": self.processed_pdfs,
            "failed_pdfs": self.failed_pdfs,
            "successful_pdfs": self.successful_pdfs,
            "current_pdf": self.current_pdf,
            "start_time": self.start_time.isoformat(),
            "last_checkpoint": self.last_checkpoint.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingState":
        """Create from dictionary."""
        state = cls(total_pdfs=data["total_pdfs"])
        state.processed_pdfs = data["processed_pdfs"]
        state.failed_pdfs = data["failed_pdfs"]
        state.successful_pdfs = data["successful_pdfs"]
        state.current_pdf = data.get("current_pdf")
        state.start_time = datetime.fromisoformat(data["start_time"])
        state.last_checkpoint = datetime.fromisoformat(data["last_checkpoint"])
        return state