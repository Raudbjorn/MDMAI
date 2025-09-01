"""Content enrichment module for updating character generation models with extracted content.

This module takes extracted TTRPG content and enriches the character generation
models with new traits, backgrounds, motivations, and other elements.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.character_generation.models import (
    Backstory,
    Character,
    CharacterClass,
    CharacterRace,
    ExtendedCharacter,
    NPC,
    NPCRole,
    PersonalityTrait,
    TTRPGGenre,
)
from src.ebook_extraction import ExtractedContent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CharacterTraitCategory(Enum):
    """Categories for character traits."""
    
    PHYSICAL = "physical"
    MENTAL = "mental"
    EMOTIONAL = "emotional"
    SOCIAL = "social"
    MORAL = "moral"
    BEHAVIORAL = "behavioral"
    QUIRK = "quirk"


class BackgroundCategory(Enum):
    """Categories for character backgrounds."""
    
    OCCUPATION = "occupation"
    SOCIAL_STATUS = "social_status"
    EDUCATION = "education"
    FAMILY = "family"
    MILITARY = "military"
    CRIMINAL = "criminal"
    RELIGIOUS = "religious"
    ARCANE = "arcane"
    TECHNICAL = "technical"


class MotivationCategory(Enum):
    """Categories for character motivations."""
    
    POWER = "power"
    WEALTH = "wealth"
    KNOWLEDGE = "knowledge"
    JUSTICE = "justice"
    REVENGE = "revenge"
    LOVE = "love"
    SURVIVAL = "survival"
    ADVENTURE = "adventure"
    REDEMPTION = "redemption"
    LEGACY = "legacy"


@dataclass
class EnrichedTrait:
    """Enhanced character trait with category and metadata."""
    
    trait: str
    category: CharacterTraitCategory
    description: Optional[str] = None
    genre_tags: List[TTRPGGenre] = field(default_factory=list)
    intensity: str = "moderate"  # mild, moderate, extreme
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trait": self.trait,
            "category": self.category.value,
            "description": self.description,
            "genre_tags": [g.value for g in self.genre_tags],
            "intensity": self.intensity
        }


@dataclass
class EnrichedBackground:
    """Enhanced character background with category and details."""
    
    background: str
    category: BackgroundCategory
    description: Optional[str] = None
    skills: List[str] = field(default_factory=list)
    equipment: List[str] = field(default_factory=list)
    social_standing: str = "common"  # common, noble, outcast, respected, feared
    genre_tags: List[TTRPGGenre] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "background": self.background,
            "category": self.category.value,
            "description": self.description,
            "skills": self.skills,
            "equipment": self.equipment,
            "social_standing": self.social_standing,
            "genre_tags": [g.value for g in self.genre_tags]
        }


@dataclass
class EnrichedMotivation:
    """Enhanced character motivation with category and depth."""
    
    motivation: str
    category: MotivationCategory
    description: Optional[str] = None
    strength: str = "strong"  # weak, moderate, strong, overwhelming
    is_public: bool = True  # Whether the character openly admits this motivation
    genre_tags: List[TTRPGGenre] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "motivation": self.motivation,
            "category": self.category.value,
            "description": self.description,
            "strength": self.strength,
            "is_public": self.is_public,
            "genre_tags": [g.value for g in self.genre_tags]
        }


@dataclass
class StoryHook:
    """Story hook for adventures and quests."""
    
    title: str
    description: str
    hook_type: str  # quest, mystery, conflict, discovery, rescue
    difficulty: str  # easy, moderate, hard, deadly
    genre_tags: List[TTRPGGenre] = field(default_factory=list)
    required_elements: List[str] = field(default_factory=list)  # NPCs, locations, items needed
    potential_rewards: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "description": self.description,
            "hook_type": self.hook_type,
            "difficulty": self.difficulty,
            "genre_tags": [g.value for g in self.genre_tags],
            "required_elements": self.required_elements,
            "potential_rewards": self.potential_rewards
        }


@dataclass
class WorldElement:
    """World-building element for campaigns."""
    
    name: str
    element_type: str  # location, faction, culture, technology, magic
    description: str
    genre_tags: List[TTRPGGenre] = field(default_factory=list)
    associated_npcs: List[str] = field(default_factory=list)
    associated_items: List[str] = field(default_factory=list)
    plot_importance: str = "minor"  # minor, moderate, major, critical
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "element_type": self.element_type,
            "description": self.description,
            "genre_tags": [g.value for g in self.genre_tags],
            "associated_npcs": self.associated_npcs,
            "associated_items": self.associated_items,
            "plot_importance": self.plot_importance
        }


@dataclass
class EnrichedContent:
    """Container for all enriched TTRPG content."""
    
    traits: List[EnrichedTrait] = field(default_factory=list)
    backgrounds: List[EnrichedBackground] = field(default_factory=list)
    motivations: List[EnrichedMotivation] = field(default_factory=list)
    story_hooks: List[StoryHook] = field(default_factory=list)
    world_elements: List[WorldElement] = field(default_factory=list)
    
    # Direct lists for simpler elements
    personality_traits: List[str] = field(default_factory=list)
    ideals: List[str] = field(default_factory=list)
    bonds: List[str] = field(default_factory=list)
    flaws: List[str] = field(default_factory=list)
    fears: List[str] = field(default_factory=list)
    
    npc_archetypes: List[str] = field(default_factory=list)
    npc_personalities: List[str] = field(default_factory=list)
    npc_quirks: List[str] = field(default_factory=list)
    
    character_names: List[str] = field(default_factory=list)
    place_names: List[str] = field(default_factory=list)
    organization_names: List[str] = field(default_factory=list)
    
    weapons: List[str] = field(default_factory=list)
    armor: List[str] = field(default_factory=list)
    items: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "traits": [t.to_dict() for t in self.traits],
            "backgrounds": [b.to_dict() for b in self.backgrounds],
            "motivations": [m.to_dict() for m in self.motivations],
            "story_hooks": [s.to_dict() for s in self.story_hooks],
            "world_elements": [w.to_dict() for w in self.world_elements],
            "personality_traits": self.personality_traits,
            "ideals": self.ideals,
            "bonds": self.bonds,
            "flaws": self.flaws,
            "fears": self.fears,
            "npc_archetypes": self.npc_archetypes,
            "npc_personalities": self.npc_personalities,
            "npc_quirks": self.npc_quirks,
            "character_names": self.character_names,
            "place_names": self.place_names,
            "organization_names": self.organization_names,
            "weapons": self.weapons,
            "armor": self.armor,
            "items": self.items
        }
    
    def save(self, output_path: Path) -> None:
        """Save enriched content to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved enriched content to {output_path}")
    
    @classmethod
    def load(cls, input_path: Path) -> 'EnrichedContent':
        """Load enriched content from JSON file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        content = cls()
        
        # Load complex objects
        for trait_data in data.get("traits", []):
            trait = EnrichedTrait(
                trait=trait_data["trait"],
                category=CharacterTraitCategory(trait_data["category"]),
                description=trait_data.get("description"),
                genre_tags=[TTRPGGenre(g) for g in trait_data.get("genre_tags", [])],
                intensity=trait_data.get("intensity", "moderate")
            )
            content.traits.append(trait)
        
        for bg_data in data.get("backgrounds", []):
            background = EnrichedBackground(
                background=bg_data["background"],
                category=BackgroundCategory(bg_data["category"]),
                description=bg_data.get("description"),
                skills=bg_data.get("skills", []),
                equipment=bg_data.get("equipment", []),
                social_standing=bg_data.get("social_standing", "common"),
                genre_tags=[TTRPGGenre(g) for g in bg_data.get("genre_tags", [])]
            )
            content.backgrounds.append(background)
        
        for mot_data in data.get("motivations", []):
            motivation = EnrichedMotivation(
                motivation=mot_data["motivation"],
                category=MotivationCategory(mot_data["category"]),
                description=mot_data.get("description"),
                strength=mot_data.get("strength", "strong"),
                is_public=mot_data.get("is_public", True),
                genre_tags=[TTRPGGenre(g) for g in mot_data.get("genre_tags", [])]
            )
            content.motivations.append(motivation)
        
        # Load simple lists
        for field in ["personality_traits", "ideals", "bonds", "flaws", "fears",
                     "npc_archetypes", "npc_personalities", "npc_quirks",
                     "character_names", "place_names", "organization_names",
                     "weapons", "armor", "items"]:
            setattr(content, field, data.get(field, []))
        
        return content


class ContentEnricher:
    """Enriches extracted content with categories and metadata."""
    
    def __init__(self):
        """Initialize the content enricher."""
        self.genre_detector = GenreDetector()
    
    def enrich_extracted_content(self, extracted: ExtractedContent) -> EnrichedContent:
        """Enrich extracted content with categories and metadata."""
        enriched = EnrichedContent()
        
        # Enrich character traits
        for trait in extracted.character_traits:
            enriched_trait = self._categorize_trait(trait)
            if enriched_trait:
                enriched.traits.append(enriched_trait)
        
        # Enrich backgrounds
        for background in extracted.backgrounds:
            enriched_bg = self._categorize_background(background)
            if enriched_bg:
                enriched.backgrounds.append(enriched_bg)
        
        # Enrich motivations
        for motivation in extracted.motivations:
            enriched_mot = self._categorize_motivation(motivation)
            if enriched_mot:
                enriched.motivations.append(enriched_mot)
        
        # Copy simple lists directly
        enriched.personality_traits = sorted(list(extracted.character_traits))[:100]
        enriched.ideals = sorted(list(extracted.ideals))[:50]
        enriched.bonds = sorted(list(extracted.bonds))[:50]
        enriched.flaws = sorted(list(extracted.flaws))[:50]
        enriched.fears = sorted(list(extracted.fears))[:50]
        
        enriched.npc_archetypes = sorted(list(extracted.npc_archetypes))[:50]
        enriched.npc_personalities = sorted(list(extracted.npc_personalities))[:50]
        enriched.npc_quirks = sorted(list(extracted.npc_quirks))[:50]
        
        enriched.character_names = sorted(list(extracted.character_names))[:200]
        enriched.place_names = sorted(list(extracted.place_names))[:100]
        enriched.organization_names = sorted(list(extracted.organization_names))[:50]
        
        enriched.weapons = sorted(list(extracted.weapons))[:100]
        enriched.armor = sorted(list(extracted.armor))[:50]
        enriched.items = sorted(list(extracted.items))[:100]
        
        # Create story hooks from extracted content
        enriched.story_hooks = self._generate_story_hooks(extracted)
        
        # Create world elements
        enriched.world_elements = self._generate_world_elements(extracted)
        
        return enriched
    
    def _categorize_trait(self, trait: str) -> Optional[EnrichedTrait]:
        """Categorize a character trait."""
        trait = trait.strip()
        if not trait or len(trait) < 3:
            return None
        
        # Determine category based on keywords
        lower_trait = trait.lower()
        
        if any(word in lower_trait for word in ['strong', 'weak', 'tall', 'short', 'fast', 'slow']):
            category = CharacterTraitCategory.PHYSICAL
        elif any(word in lower_trait for word in ['smart', 'clever', 'intelligent', 'wise', 'foolish']):
            category = CharacterTraitCategory.MENTAL
        elif any(word in lower_trait for word in ['happy', 'sad', 'angry', 'calm', 'nervous']):
            category = CharacterTraitCategory.EMOTIONAL
        elif any(word in lower_trait for word in ['friendly', 'hostile', 'charming', 'rude']):
            category = CharacterTraitCategory.SOCIAL
        elif any(word in lower_trait for word in ['honest', 'loyal', 'evil', 'good', 'corrupt']):
            category = CharacterTraitCategory.MORAL
        elif any(word in lower_trait for word in ['aggressive', 'passive', 'cautious', 'reckless']):
            category = CharacterTraitCategory.BEHAVIORAL
        else:
            category = CharacterTraitCategory.QUIRK
        
        # Detect genre
        genres = self.genre_detector.detect_genres(trait)
        
        return EnrichedTrait(
            trait=trait,
            category=category,
            genre_tags=genres
        )
    
    def _categorize_background(self, background: str) -> Optional[EnrichedBackground]:
        """Categorize a character background."""
        background = background.strip()
        if not background or len(background) < 3:
            return None
        
        lower_bg = background.lower()
        
        # Determine category
        if any(word in lower_bg for word in ['soldier', 'warrior', 'captain', 'general']):
            category = BackgroundCategory.MILITARY
        elif any(word in lower_bg for word in ['thief', 'criminal', 'smuggler', 'pirate']):
            category = BackgroundCategory.CRIMINAL
        elif any(word in lower_bg for word in ['priest', 'cleric', 'monk', 'paladin']):
            category = BackgroundCategory.RELIGIOUS
        elif any(word in lower_bg for word in ['wizard', 'mage', 'sorcerer', 'witch']):
            category = BackgroundCategory.ARCANE
        elif any(word in lower_bg for word in ['engineer', 'scientist', 'hacker', 'tech']):
            category = BackgroundCategory.TECHNICAL
        elif any(word in lower_bg for word in ['noble', 'lord', 'lady', 'prince']):
            category = BackgroundCategory.SOCIAL_STATUS
        elif any(word in lower_bg for word in ['scholar', 'student', 'professor', 'teacher']):
            category = BackgroundCategory.EDUCATION
        else:
            category = BackgroundCategory.OCCUPATION
        
        # Detect genre
        genres = self.genre_detector.detect_genres(background)
        
        return EnrichedBackground(
            background=background,
            category=category,
            genre_tags=genres
        )
    
    def _categorize_motivation(self, motivation: str) -> Optional[EnrichedMotivation]:
        """Categorize a character motivation."""
        motivation = motivation.strip()
        if not motivation or len(motivation) < 5:
            return None
        
        lower_mot = motivation.lower()
        
        # Determine category
        if any(word in lower_mot for word in ['power', 'control', 'rule', 'dominate']):
            category = MotivationCategory.POWER
        elif any(word in lower_mot for word in ['money', 'wealth', 'rich', 'gold', 'treasure']):
            category = MotivationCategory.WEALTH
        elif any(word in lower_mot for word in ['knowledge', 'learn', 'discover', 'understand']):
            category = MotivationCategory.KNOWLEDGE
        elif any(word in lower_mot for word in ['justice', 'right', 'fair', 'law']):
            category = MotivationCategory.JUSTICE
        elif any(word in lower_mot for word in ['revenge', 'vengeance', 'payback', 'retribution']):
            category = MotivationCategory.REVENGE
        elif any(word in lower_mot for word in ['love', 'romance', 'heart', 'affection']):
            category = MotivationCategory.LOVE
        elif any(word in lower_mot for word in ['survive', 'live', 'escape', 'safety']):
            category = MotivationCategory.SURVIVAL
        elif any(word in lower_mot for word in ['adventure', 'explore', 'journey', 'quest']):
            category = MotivationCategory.ADVENTURE
        else:
            category = MotivationCategory.LEGACY
        
        # Detect genre
        genres = self.genre_detector.detect_genres(motivation)
        
        return EnrichedMotivation(
            motivation=motivation[:100],  # Limit length
            category=category,
            genre_tags=genres
        )
    
    def _generate_story_hooks(self, extracted: ExtractedContent) -> List[StoryHook]:
        """Generate story hooks from extracted content."""
        hooks = []
        
        # Create hooks from quest ideas
        for quest in list(extracted.quest_ideas)[:20]:
            hook = StoryHook(
                title=quest[:50],
                description=quest,
                hook_type="quest",
                difficulty="moderate",
                genre_tags=self.genre_detector.detect_genres(quest)
            )
            hooks.append(hook)
        
        # Create hooks from conflicts
        for conflict in list(extracted.conflicts)[:10]:
            hook = StoryHook(
                title=f"Conflict: {conflict[:30]}",
                description=conflict,
                hook_type="conflict",
                difficulty="hard",
                genre_tags=self.genre_detector.detect_genres(conflict)
            )
            hooks.append(hook)
        
        return hooks
    
    def _generate_world_elements(self, extracted: ExtractedContent) -> List[WorldElement]:
        """Generate world elements from extracted content."""
        elements = []
        
        # Create location elements
        for location in list(extracted.locations)[:30]:
            element = WorldElement(
                name=location,
                element_type="location",
                description=f"A location in the game world",
                genre_tags=self.genre_detector.detect_genres(location)
            )
            elements.append(element)
        
        # Create faction elements
        for faction in list(extracted.factions)[:20]:
            element = WorldElement(
                name=faction,
                element_type="faction",
                description=f"An organization or group",
                genre_tags=self.genre_detector.detect_genres(faction)
            )
            elements.append(element)
        
        return elements


class GenreDetector:
    """Detect TTRPG genre from text content."""
    
    # Genre keyword mappings
    genre_keywords = {
        TTRPGGenre.FANTASY: [
            'magic', 'wizard', 'dragon', 'elf', 'dwarf', 'sword', 'spell',
            'castle', 'kingdom', 'quest', 'dungeon', 'orc', 'goblin'
        ],
        TTRPGGenre.SCI_FI: [
            'space', 'ship', 'laser', 'alien', 'planet', 'galaxy', 'star',
            'android', 'robot', 'cyborg', 'quantum', 'warp', 'station'
        ],
        TTRPGGenre.CYBERPUNK: [
            'cyber', 'hack', 'net', 'corp', 'chrome', 'street', 'data',
            'neural', 'implant', 'matrix', 'runner', 'punk', 'neon'
        ],
        TTRPGGenre.COSMIC_HORROR: [
            'elder', 'ancient', 'madness', 'sanity', 'cosmic', 'cult',
            'forbidden', 'tentacle', 'void', 'darkness', 'horror'
        ],
        TTRPGGenre.POST_APOCALYPTIC: [
            'wasteland', 'radiation', 'mutant', 'survivor', 'vault',
            'scavenger', 'ruins', 'apocalypse', 'fallout', 'raider'
        ],
        TTRPGGenre.WESTERN: [
            'gunslinger', 'sheriff', 'saloon', 'outlaw', 'ranch',
            'frontier', 'cowboy', 'desert', 'duel', 'marshal'
        ],
        TTRPGGenre.SUPERHERO: [
            'hero', 'villain', 'power', 'super', 'cape', 'mask',
            'vigilante', 'justice', 'sidekick', 'nemesis'
        ]
    }
    
    def __init__(self):
        """Initialize genre detector."""
        pass
    
    def detect_genres(self, text: str) -> List[TTRPGGenre]:
        """Detect genres from text."""
        if not text:
            return []
        
        lower_text = text.lower()
        detected_genres = []
        
        for genre, keywords in self.genre_keywords.items():
            if any(keyword in lower_text for keyword in keywords):
                detected_genres.append(genre)
        
        # Default to fantasy if no genre detected
        if not detected_genres:
            detected_genres.append(TTRPGGenre.FANTASY)
        
        return detected_genres[:2]  # Limit to 2 most relevant genres


def main():
    """Main function to enrich extracted content."""
    # Load extracted content
    extracted_path = Path("extracted_content/ttrpg_content.json")
    
    if not extracted_path.exists():
        logger.error(f"Extracted content not found at {extracted_path}")
        logger.info("Please run ebook_extraction.py first")
        return None
    
    # Load the extracted content
    from ebook_extraction import EbookExtractor
    extractor = EbookExtractor()
    extracted_content = extractor.load_extracted_content(extracted_path)
    
    # Enrich the content
    enricher = ContentEnricher()
    enriched_content = enricher.enrich_extracted_content(extracted_content)
    
    # Save enriched content
    output_path = Path("extracted_content/enriched_content.json")
    enriched_content.save(output_path)
    
    # Print summary
    logger.info("\n=== Enrichment Summary ===")
    logger.info(f"Enriched Traits: {len(enriched_content.traits)}")
    logger.info(f"Enriched Backgrounds: {len(enriched_content.backgrounds)}")
    logger.info(f"Enriched Motivations: {len(enriched_content.motivations)}")
    logger.info(f"Story Hooks: {len(enriched_content.story_hooks)}")
    logger.info(f"World Elements: {len(enriched_content.world_elements)}")
    
    return enriched_content


if __name__ == "__main__":
    enriched = main()