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


class TTRPGGenre(Enum):
    """Enumeration of TTRPG genres for classification."""
    FANTASY = auto()
    SCI_FI = auto()
    CYBERPUNK = auto()
    COSMIC_HORROR = auto()
    POST_APOCALYPTIC = auto()
    STEAMPUNK = auto()
    URBAN_FANTASY = auto()
    SPACE_OPERA = auto()
    SUPERHERO = auto()
    HISTORICAL = auto()
    WESTERN = auto()
    NOIR = auto()
    PULP = auto()
    MODERN = auto()
    MILITARY = auto()
    HORROR = auto()
    MYSTERY = auto()
    MYTHOLOGICAL = auto()
    ANIME = auto()
    GENERIC = auto()
    UNKNOWN = auto()


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