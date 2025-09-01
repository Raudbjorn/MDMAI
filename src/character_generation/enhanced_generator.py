"""Enhanced character generator that uses enriched content from ebook extraction.

This module extends the base character generator with rich content extracted
from TTRPG ebooks, providing more diverse and interesting character options.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .character_generator import CharacterGenerator
from .models import (
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

logger = logging.getLogger(__name__)


class EnrichedContentManager:
    """Manages and provides access to enriched TTRPG content."""
    
    def __init__(self, content_path: Optional[Path] = None):
        """Initialize the content manager.
        
        Args:
            content_path: Path to the enriched content JSON file.
                         Defaults to extracted_content/enriched_content.json
        """
        self.content_path = content_path or Path("extracted_content/enriched_content.json")
        self.content = self._load_content()
        
        # Cache frequently used content
        self._trait_cache = {}
        self._background_cache = {}
        self._motivation_cache = {}
    
    def _load_content(self) -> Dict[str, Any]:
        """Load enriched content from file."""
        if not self.content_path.exists():
            logger.warning(f"Enriched content not found at {self.content_path}")
            return self._get_default_content()
        
        try:
            with open(self.content_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading enriched content: {e}")
            return self._get_default_content()
    
    def _get_default_content(self) -> Dict[str, Any]:
        """Get default content if enriched content is not available."""
        return {
            "personality_traits": [
                "Brave", "Cautious", "Curious", "Determined", "Friendly",
                "Honest", "Impulsive", "Loyal", "Mysterious", "Optimistic",
                "Pessimistic", "Quiet", "Rebellious", "Serious", "Witty"
            ],
            "ideals": [
                "Freedom", "Justice", "Power", "Knowledge", "Honor",
                "Wealth", "Family", "Tradition", "Change", "Balance"
            ],
            "bonds": [
                "Family legacy", "Lost love", "Sacred oath", "Childhood friend",
                "Mentor's teachings", "Home village", "Ancient artifact", "Debt of honor"
            ],
            "flaws": [
                "Overconfident", "Greedy", "Cowardly", "Arrogant", "Naive",
                "Hot-tempered", "Distrustful", "Reckless", "Indecisive", "Vengeful"
            ],
            "fears": [
                "Death", "Darkness", "Heights", "Water", "Fire",
                "Betrayal", "Failure", "Loneliness", "Magic", "The unknown"
            ],
            "backgrounds": [],
            "motivations": [],
            "character_names": [],
            "weapons": [],
            "armor": [],
            "items": []
        }
    
    def get_traits_for_genre(self, genre: TTRPGGenre, count: int = 3) -> List[str]:
        """Get personality traits suitable for a specific genre."""
        # Check if we have genre-specific traits
        genre_traits = []
        
        if "traits" in self.content:
            for trait_data in self.content["traits"]:
                if genre.value in [g for g in trait_data.get("genre_tags", [])]:
                    genre_traits.append(trait_data["trait"])
        
        # Fallback to general traits
        if not genre_traits:
            genre_traits = self.content.get("personality_traits", [])
        
        # Return random selection
        if len(genre_traits) >= count:
            return random.sample(genre_traits, count)
        return genre_traits
    
    def get_background_for_genre(self, genre: TTRPGGenre) -> Optional[str]:
        """Get a background suitable for a specific genre."""
        genre_backgrounds = []
        
        if "backgrounds" in self.content:
            for bg_data in self.content["backgrounds"]:
                if genre.value in [g for g in bg_data.get("genre_tags", [])]:
                    genre_backgrounds.append(bg_data["background"])
        
        if genre_backgrounds:
            return random.choice(genre_backgrounds)
        
        # Fallback to any background
        all_backgrounds = [bg["background"] for bg in self.content.get("backgrounds", [])]
        if all_backgrounds:
            return random.choice(all_backgrounds)
        
        return None
    
    def get_motivation_for_genre(self, genre: TTRPGGenre) -> Optional[str]:
        """Get a motivation suitable for a specific genre."""
        genre_motivations = []
        
        if "motivations" in self.content:
            for mot_data in self.content["motivations"]:
                if genre.value in [g for g in mot_data.get("genre_tags", [])]:
                    genre_motivations.append(mot_data["motivation"])
        
        if genre_motivations:
            return random.choice(genre_motivations)
        
        # Fallback to any motivation
        all_motivations = [mot["motivation"] for mot in self.content.get("motivations", [])]
        if all_motivations:
            return random.choice(all_motivations)
        
        return "Seeking adventure and fortune"
    
    def get_random_name(self, name_type: str = "character") -> str:
        """Get a random name from extracted content."""
        if name_type == "character":
            names = self.content.get("character_names", [])
        elif name_type == "place":
            names = self.content.get("place_names", [])
        elif name_type == "organization":
            names = self.content.get("organization_names", [])
        else:
            names = []
        
        if names:
            return random.choice(names)
        
        # Fallback name generation
        prefixes = ["Dar", "Kor", "Val", "Zar", "Mor", "Tal", "Kal", "Sar"]
        suffixes = ["ian", "ius", "on", "ar", "eth", "in", "an", "os"]
        return random.choice(prefixes) + random.choice(suffixes)
    
    def get_equipment_set(self, equipment_type: str, count: int = 3) -> List[str]:
        """Get a set of equipment items."""
        if equipment_type == "weapons":
            items = self.content.get("weapons", [])
        elif equipment_type == "armor":
            items = self.content.get("armor", [])
        else:
            items = self.content.get("items", [])
        
        if len(items) >= count:
            return random.sample(items, count)
        return items
    
    def get_story_hook(self, genre: Optional[TTRPGGenre] = None) -> Optional[Dict[str, Any]]:
        """Get a story hook, optionally filtered by genre."""
        hooks = self.content.get("story_hooks", [])
        
        if genre and hooks:
            genre_hooks = [
                hook for hook in hooks 
                if genre.value in hook.get("genre_tags", [])
            ]
            if genre_hooks:
                return random.choice(genre_hooks)
        
        if hooks:
            return random.choice(hooks)
        
        return None
    
    def get_world_element(self, element_type: str, genre: Optional[TTRPGGenre] = None) -> Optional[Dict[str, Any]]:
        """Get a world-building element."""
        elements = self.content.get("world_elements", [])
        
        # Filter by type
        typed_elements = [
            elem for elem in elements 
            if elem.get("element_type") == element_type
        ]
        
        # Further filter by genre if specified
        if genre and typed_elements:
            genre_elements = [
                elem for elem in typed_elements
                if genre.value in elem.get("genre_tags", [])
            ]
            if genre_elements:
                return random.choice(genre_elements)
        
        if typed_elements:
            return random.choice(typed_elements)
        
        return None


class EnhancedCharacterGenerator(CharacterGenerator):
    """Enhanced character generator with enriched content support."""
    
    def __init__(self, content_manager: Optional[EnrichedContentManager] = None):
        """Initialize the enhanced generator.
        
        Args:
            content_manager: Manager for enriched content. Creates default if None.
        """
        super().__init__()
        self.content_manager = content_manager or EnrichedContentManager()
    
    def generate_character(
        self,
        system: str = "D&D 5e",
        genre: Optional[TTRPGGenre] = None,
        character_class: Optional[CharacterClass] = None,
        race: Optional[CharacterRace] = None,
        level: int = 1,
        use_enriched_content: bool = True,
    ) -> ExtendedCharacter:
        """Generate a character with enriched content.
        
        Args:
            system: The TTRPG system (e.g., "D&D 5e", "Pathfinder")
            genre: The genre of the campaign
            character_class: Specific class to use (random if None)
            race: Specific race to use (random if None)
            level: Character level
            use_enriched_content: Whether to use enriched content
            
        Returns:
            ExtendedCharacter with enriched backstory and details
        """
        # Set default genre if not provided
        if genre is None:
            genre = TTRPGGenre.FANTASY
        
        # Generate base character using parent class
        base_char = super().generate_character(
            system=system,
            character_class=character_class,
            race=race,
            level=level
        )
        
        # Convert to ExtendedCharacter
        character = ExtendedCharacter(**base_char.__dict__)
        character.genre = genre
        
        if use_enriched_content:
            # Enhance with enriched content
            self._enhance_with_enriched_content(character, genre)
        
        return character
    
    def _enhance_with_enriched_content(self, character: ExtendedCharacter, genre: TTRPGGenre) -> None:
        """Enhance character with enriched content."""
        # Set enriched name
        name = self.content_manager.get_random_name("character")
        if name:
            character.name = name
        
        # Enhance backstory with enriched content
        backstory = character.backstory
        
        # Add enriched personality traits
        traits = self.content_manager.get_traits_for_genre(genre, count=3)
        backstory.personality_traits.extend(traits)
        
        # Add enriched background
        background = self.content_manager.get_background_for_genre(genre)
        if background:
            backstory.background = background
        
        # Add enriched motivation
        motivation = self.content_manager.get_motivation_for_genre(genre)
        if motivation:
            backstory.motivation = motivation
        
        # Add enriched ideals, bonds, and flaws
        if self.content_manager.content.get("ideals"):
            backstory.ideals = random.sample(
                self.content_manager.content["ideals"],
                min(2, len(self.content_manager.content["ideals"]))
            )
        
        if self.content_manager.content.get("bonds"):
            backstory.bonds = random.sample(
                self.content_manager.content["bonds"],
                min(2, len(self.content_manager.content["bonds"]))
            )
        
        if self.content_manager.content.get("flaws"):
            backstory.flaws = random.sample(
                self.content_manager.content["flaws"],
                min(2, len(self.content_manager.content["flaws"]))
            )
        
        if self.content_manager.content.get("fears"):
            backstory.fears = random.sample(
                self.content_manager.content["fears"],
                min(2, len(self.content_manager.content["fears"]))
            )
        
        # Add enriched equipment if available
        weapons = self.content_manager.get_equipment_set("weapons", count=2)
        if weapons:
            character.equipment.weapons.extend(weapons)
        
        armor = self.content_manager.get_equipment_set("armor", count=1)
        if armor:
            character.equipment.armor.extend(armor)
        
        items = self.content_manager.get_equipment_set("items", count=3)
        if items:
            character.equipment.items.extend(items)
        
        # Add goals based on story hooks
        story_hook = self.content_manager.get_story_hook(genre)
        if story_hook:
            backstory.goals.append(story_hook.get("title", "Unknown quest"))
        
        # Add location-based origin
        location = self.content_manager.get_world_element("location", genre)
        if location:
            backstory.origin = f"Originally from {location.get('name', 'unknown lands')}"
    
    def generate_npc_with_enriched_content(
        self,
        role: Optional[NPCRole] = None,
        genre: Optional[TTRPGGenre] = None,
        importance: str = "Minor",
    ) -> NPC:
        """Generate an NPC with enriched content.
        
        Args:
            role: NPC role (random if None)
            genre: Campaign genre
            importance: NPC importance level
            
        Returns:
            NPC with enriched details
        """
        # Set defaults
        if genre is None:
            genre = TTRPGGenre.FANTASY
        
        if role is None:
            # Choose a random role appropriate for the genre
            role = self._get_random_role_for_genre(genre)
        
        # Generate base NPC
        npc = NPC()
        npc.role = role
        npc.genre = genre
        npc.importance = importance
        
        # Set enriched name
        name = self.content_manager.get_random_name("character")
        if name:
            npc.name = name
        
        # Add personality traits from enriched content
        traits = self.content_manager.get_traits_for_genre(genre, count=2)
        for trait in traits:
            npc.personality_traits.append(
                PersonalityTrait(
                    category="personality",
                    trait=trait,
                    description=f"This character is notably {trait.lower()}"
                )
            )
        
        # Add quirks
        if self.content_manager.content.get("npc_quirks"):
            quirks = random.sample(
                self.content_manager.content["npc_quirks"],
                min(1, len(self.content_manager.content["npc_quirks"]))
            )
            for quirk in quirks:
                npc.personality_traits.append(
                    PersonalityTrait(
                        category="quirk",
                        trait=quirk,
                        description="A distinctive quirk"
                    )
                )
        
        # Set background
        background = self.content_manager.get_background_for_genre(genre)
        if background:
            npc.backstory.background = background
        
        # Set motivation
        motivation = self.content_manager.get_motivation_for_genre(genre)
        if motivation:
            npc.backstory.motivation = motivation
        
        # Add fears
        if self.content_manager.content.get("fears"):
            npc.backstory.fears = random.sample(
                self.content_manager.content["fears"],
                min(1, len(self.content_manager.content["fears"]))
            )
        
        # Set location
        location = self.content_manager.get_world_element("location", genre)
        if location:
            npc.location = location.get("name", "Unknown location")
        
        # Set faction if available
        faction = self.content_manager.get_world_element("faction", genre)
        if faction:
            npc.faction = faction.get("name")
        
        # Add knowledge areas based on role and background
        npc.knowledge_areas = self._generate_knowledge_areas(npc.role, genre)
        
        # Add secrets for important NPCs
        if importance in ["Supporting", "Major"]:
            npc.secrets = self._generate_secrets(genre)
        
        # Generate basic stats
        # Use a default class for NPCs based on their role
        npc_class = CharacterClass.FIGHTER  # Default class for NPCs
        npc_race = CharacterRace.HUMAN  # Default race for NPCs
        npc.stats = self._generate_stats(
            method="standard",
            character_class=npc_class,
            race=npc_race,
            level=1
        )
        
        # Add equipment
        if npc.role in [NPCRole.GUARD, NPCRole.SOLDIER, NPCRole.ADVENTURER]:
            weapons = self.content_manager.get_equipment_set("weapons", count=1)
            if weapons:
                npc.equipment.weapons.extend(weapons)
            armor = self.content_manager.get_equipment_set("armor", count=1)
            if armor:
                npc.equipment.armor.extend(armor)
        
        items = self.content_manager.get_equipment_set("items", count=2)
        if items:
            npc.equipment.items.extend(items)
        
        return npc
    
    def _get_random_role_for_genre(self, genre: TTRPGGenre) -> NPCRole:
        """Get a random NPC role appropriate for the genre."""
        genre_roles = {
            TTRPGGenre.FANTASY: [
                NPCRole.MERCHANT, NPCRole.GUARD, NPCRole.INNKEEPER,
                NPCRole.SCHOLAR, NPCRole.PRIEST, NPCRole.ADVENTURER
            ],
            TTRPGGenre.SCI_FI: [
                NPCRole.STATION_COMMANDER, NPCRole.XENOBIOLOGIST_NPC,
                NPCRole.SPACE_TRADER, NPCRole.SMUGGLER
            ],
            TTRPGGenre.CYBERPUNK: [
                NPCRole.STREET_SAMURAI, NPCRole.CORPORATE_EXEC,
                NPCRole.INFO_BROKER, NPCRole.RIPPERDOC
            ],
            TTRPGGenre.COSMIC_HORROR: [
                NPCRole.CULTIST, NPCRole.LIBRARIAN, NPCRole.PRIVATE_INVESTIGATOR,
                NPCRole.MUSEUM_CURATOR
            ],
            TTRPGGenre.POST_APOCALYPTIC: [
                NPCRole.SCRAP_DEALER, NPCRole.WASTELAND_DOCTOR,
                NPCRole.CARAVAN_MASTER, NPCRole.TRIBAL_ELDER
            ],
            TTRPGGenre.WESTERN: [
                NPCRole.SHERIFF, NPCRole.SALOON_KEEPER,
                NPCRole.BLACKSMITH, NPCRole.BANK_TELLER
            ],
        }
        
        roles = genre_roles.get(genre, [NPCRole.COMMONER])
        return random.choice(roles)
    
    def _generate_knowledge_areas(self, role: NPCRole, genre: TTRPGGenre) -> List[str]:
        """Generate knowledge areas based on role and genre."""
        knowledge_map = {
            NPCRole.SCHOLAR: ["History", "Ancient languages", "Arcane theory"],
            NPCRole.MERCHANT: ["Trade routes", "Market prices", "Foreign goods"],
            NPCRole.GUARD: ["Local laws", "City layout", "Known criminals"],
            NPCRole.PRIEST: ["Religious doctrine", "Healing arts", "Divine magic"],
            NPCRole.LIBRARIAN: ["Forbidden knowledge", "Ancient texts", "Research methods"],
            NPCRole.INFO_BROKER: ["Street rumors", "Corporate secrets", "Underground networks"],
            NPCRole.XENOBIOLOGIST_NPC: ["Alien species", "Exobiology", "Terraforming"],
        }
        
        return knowledge_map.get(role, ["Local area", "Common knowledge"])
    
    def _generate_secrets(self, genre: TTRPGGenre) -> List[str]:
        """Generate secrets based on genre."""
        genre_secrets = {
            TTRPGGenre.FANTASY: [
                "Knows the location of a hidden treasure",
                "Is secretly a member of a thieves' guild",
                "Has royal blood but hides it",
                "Possesses a cursed artifact"
            ],
            TTRPGGenre.SCI_FI: [
                "Is an undercover alien spy",
                "Knows coordinates to a derelict ship",
                "Has illegal AI implants",
                "Witnessed first contact"
            ],
            TTRPGGenre.CYBERPUNK: [
                "Is a corporate double agent",
                "Knows a backdoor into the main grid",
                "Has dirt on a powerful exec",
                "Runs illegal braindances"
            ],
            TTRPGGenre.COSMIC_HORROR: [
                "Has read the forbidden texts",
                "Knows the cult's true purpose",
                "Has seen the old ones in dreams",
                "Guards an eldritch artifact"
            ],
        }
        
        secrets = genre_secrets.get(genre, ["Has a mysterious past"])
        return random.sample(secrets, min(2, len(secrets)))
    
    def create_party_with_story_hook(
        self,
        party_size: int = 4,
        genre: Optional[TTRPGGenre] = None,
        system: str = "D&D 5e",
    ) -> Tuple[List[ExtendedCharacter], Dict[str, Any]]:
        """Create a party of characters with a shared story hook.
        
        Args:
            party_size: Number of characters in the party
            genre: Campaign genre
            system: TTRPG system
            
        Returns:
            Tuple of (character list, story hook dict)
        """
        if genre is None:
            genre = TTRPGGenre.FANTASY
        
        # Get a story hook for the party
        story_hook = self.content_manager.get_story_hook(genre)
        if not story_hook:
            story_hook = {
                "title": "The Mystery Quest",
                "description": "The party must uncover a hidden truth",
                "hook_type": "quest",
                "difficulty": "moderate"
            }
        
        # Generate party members
        party = []
        used_classes = set()
        
        for i in range(party_size):
            # Try to avoid duplicate classes
            available_classes = [
                c for c in CharacterClass 
                if c not in used_classes and c != CharacterClass.CUSTOM
            ]
            
            if available_classes:
                char_class = random.choice(available_classes)
                used_classes.add(char_class)
            else:
                char_class = None
            
            # Generate character
            character = self.generate_character(
                system=system,
                genre=genre,
                character_class=char_class,
                level=1,
                use_enriched_content=True
            )
            
            # Add the story hook to their goals
            character.backstory.goals.append(story_hook["title"])
            
            # Create a bond with another party member
            if i > 0:
                other_member = party[random.randint(0, i-1)]
                character.backstory.relationships.append({
                    "name": other_member.name,
                    "relationship": random.choice([
                        "childhood friend",
                        "former rival",
                        "saved their life",
                        "owes them a debt",
                        "shares a secret"
                    ])
                })
            
            party.append(character)
        
        return party, story_hook


def main():
    """Example usage of the enhanced character generator."""
    # Initialize the enhanced generator
    generator = EnhancedCharacterGenerator()
    
    # Generate a character with enriched content
    print("\n=== Generating Enhanced Character ===")
    character = generator.generate_character(
        genre=TTRPGGenre.FANTASY,
        character_class=CharacterClass.ROGUE,
        use_enriched_content=True
    )
    
    print(f"Name: {character.name}")
    print(f"Class: {character.get_class_name()}")
    print(f"Race: {character.get_race_name()}")
    print(f"Background: {character.backstory.background}")
    print(f"Motivation: {character.backstory.motivation}")
    print(f"Personality Traits: {', '.join(character.backstory.personality_traits[:3])}")
    print(f"Goals: {', '.join(character.backstory.goals[:2])}")
    
    # Generate an NPC with enriched content
    print("\n=== Generating Enhanced NPC ===")
    npc = generator.generate_npc_with_enriched_content(
        role=NPCRole.MERCHANT,
        genre=TTRPGGenre.FANTASY,
        importance="Supporting"
    )
    
    print(f"Name: {npc.name}")
    print(f"Role: {npc.get_role_name()}")
    print(f"Location: {npc.location}")
    print(f"Motivation: {npc.backstory.motivation}")
    print(f"Secrets: {', '.join(npc.secrets[:2])}")
    
    # Generate a party with a story hook
    print("\n=== Generating Party with Story Hook ===")
    party, story_hook = generator.create_party_with_story_hook(
        party_size=4,
        genre=TTRPGGenre.CYBERPUNK
    )
    
    print(f"Story Hook: {story_hook['title']}")
    print(f"Description: {story_hook['description']}")
    print("\nParty Members:")
    for char in party:
        print(f"  - {char.name}: {char.get_class_name()} ({char.get_race_name()})")
        if char.backstory.relationships:
            rel = char.backstory.relationships[0]
            print(f"    Relationship: {rel['relationship']} with {rel['name']}")


if __name__ == "__main__":
    main()