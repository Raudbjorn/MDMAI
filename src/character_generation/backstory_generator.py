"""Backstory generation system for TTRPG characters."""

import logging
import random
from typing import Any, Dict, List, Optional

from .models import (
    Backstory,
    Character,
    CharacterBackground,
    CharacterClass,
    CharacterMotivation,
    CharacterRace,
    CharacterTrait,
    ExtendedCharacter,
    StoryHook,
    TTRPGGenre,
    WorldElement,
)

logger = logging.getLogger(__name__)


class BackstoryGenerator:
    """Generate rich, personality-aware backstories for characters using enriched content."""
    
    # Class-level constants for trait categories
    _PHYSICAL_TRAIT_NAMES = [
        'AGILE', 'ATHLETIC', 'BRAWNY', 'BURLY', 'DELICATE', 'DEXTEROUS', 
        'ENDURING', 'ENERGETIC', 'GRACEFUL', 'HARDY', 'LITHE', 'MUSCULAR',
        'NIMBLE', 'POWERFUL', 'QUICK', 'RESILIENT', 'ROBUST', 'RUGGED',
        'SCARRED', 'SLENDER', 'STOCKY', 'STRONG', 'STURDY', 'SWIFT',
        'TALL', 'TOUGH', 'TOWERING', 'WEATHERED', 'WIRY'
    ]
    
    _MENTAL_TRAIT_NAMES = [
        'ANALYTICAL', 'ASTUTE', 'BRILLIANT', 'CALCULATING', 'CLEVER',
        'CREATIVE', 'CUNNING', 'CURIOUS', 'FOCUSED', 'GENIUS', 'IMAGINATIVE',
        'INSIGHTFUL', 'INTELLECTUAL', 'INTELLIGENT', 'INTUITIVE', 'KNOWLEDGEABLE',
        'LEARNED', 'LOGICAL', 'METHODICAL', 'OBSERVANT', 'PERCEPTIVE',
        'PHILOSOPHICAL', 'QUICK_WITTED', 'RATIONAL', 'RESOURCEFUL', 'SCHOLARLY',
        'SHARP', 'SHREWD', 'STRATEGIC', 'STUDIOUS', 'TACTICAL', 'THOUGHTFUL', 'WISE'
    ]
    
    _EMOTIONAL_TRAIT_NAMES = [
        'AMBITIOUS', 'ANXIOUS', 'BOLD', 'BRAVE', 'CALM', 'CAUTIOUS',
        'CHEERFUL', 'COMPASSIONATE', 'CONFIDENT', 'COURAGEOUS', 'DETERMINED',
        'DEVOTED', 'DISCIPLINED', 'EMPATHETIC', 'ENTHUSIASTIC', 'FEARLESS',
        'FIERCE', 'GENTLE', 'GRIM', 'HOPEFUL', 'HUMBLE', 'IMPULSIVE',
        'INDEPENDENT', 'JOYFUL', 'KIND', 'LOYAL', 'MELANCHOLIC', 'MERCIFUL',
        'PASSIONATE', 'PATIENT', 'PROUD', 'REBELLIOUS', 'RECKLESS', 'RESOLUTE',
        'RUTHLESS', 'SELFLESS', 'SERENE', 'SINCERE', 'SKEPTICAL', 'STEADFAST',
        'STOIC', 'STUBBORN', 'SYMPATHETIC', 'TENACIOUS', 'VENGEFUL', 'VIGILANT',
        'VOLATILE', 'ZEALOUS'
    ]
    
    _SOCIAL_TRAIT_NAMES = [
        'CHARISMATIC', 'CHARMING', 'DIPLOMATIC', 'ELOQUENT', 'GREGARIOUS',
        'INTIMIDATING', 'MYSTERIOUS', 'PERSUASIVE', 'RESERVED', 'SHY',
        'SOCIABLE', 'WITTY'
    ]
    
    @classmethod
    def get_random_traits(cls, count: int = 3) -> List[CharacterTrait]:
        """Get random character traits from extracted content."""
        physical_traits = [
            t for t in CharacterTrait 
            if t.name in cls._PHYSICAL_TRAIT_NAMES
        ]
        mental_traits = [
            t for t in CharacterTrait
            if t.name in cls._MENTAL_TRAIT_NAMES
        ]
        emotional_traits = [
            t for t in CharacterTrait
            if t.name in cls._EMOTIONAL_TRAIT_NAMES
        ]
        social_traits = [
            t for t in CharacterTrait
            if t.name in cls._SOCIAL_TRAIT_NAMES
        ]
        
        selected = []
        # Try to get a balanced mix of trait types
        if count >= 4:
            selected.append(random.choice(physical_traits) if physical_traits else None)
            selected.append(random.choice(mental_traits) if mental_traits else None)
            selected.append(random.choice(emotional_traits) if emotional_traits else None)
            selected.append(random.choice(social_traits) if social_traits else None)
            count -= 4
        
        # Fill remaining with random traits
        all_traits = list(CharacterTrait)
        for _ in range(count):
            trait = random.choice(all_traits)
            if trait not in selected:
                selected.append(trait)
        
        return [t for t in selected if t is not None]
    
    # Class-level constant for genre backgrounds
    _GENRE_BACKGROUNDS = {
        TTRPGGenre.FANTASY: [
            CharacterBackground.ACOLYTE, CharacterBackground.CRIMINAL,
            CharacterBackground.FOLK_HERO, CharacterBackground.NOBLE,
            CharacterBackground.SAGE, CharacterBackground.SOLDIER,
            CharacterBackground.HERMIT, CharacterBackground.ENTERTAINER,
            CharacterBackground.GUILD_ARTISAN, CharacterBackground.OUTLANDER,
            CharacterBackground.SAILOR, CharacterBackground.URCHIN,
            CharacterBackground.ALCHEMIST, CharacterBackground.KNIGHT,
            CharacterBackground.MERCHANT, CharacterBackground.MYSTIC
        ],
        TTRPGGenre.SCI_FI: [
            CharacterBackground.ASTEROID_MINER, CharacterBackground.COLONY_ADMINISTRATOR,
            CharacterBackground.CORPORATE_AGENT, CharacterBackground.CYBORG_ENGINEER,
            CharacterBackground.DATA_ANALYST, CharacterBackground.DIPLOMAT_ENVOY,
            CharacterBackground.GENETIC_RESEARCHER, CharacterBackground.HACKER,
            CharacterBackground.JUMP_PILOT, CharacterBackground.ORBITAL_MECHANIC,
            CharacterBackground.SPACE_MARINE, CharacterBackground.STARSHIP_ENGINEER,
            CharacterBackground.TERRAFORMER, CharacterBackground.VOID_TRADER,
            CharacterBackground.XENOBIOLOGIST
        ],
        TTRPGGenre.CYBERPUNK: [
            CharacterBackground.CORPORATE_EXEC, CharacterBackground.FIXER,
            CharacterBackground.GANG_MEMBER, CharacterBackground.MEDIA_JOURNALIST,
            CharacterBackground.NETRUNNER, CharacterBackground.NOMAD,
            CharacterBackground.RIPPERDOC, CharacterBackground.ROCKERBOY,
            CharacterBackground.STREET_SAMURAI, CharacterBackground.TECH_SPECIALIST
        ],
        TTRPGGenre.POST_APOCALYPTIC: [
            CharacterBackground.BUNKER_SURVIVOR, CharacterBackground.CARAVAN_TRADER,
            CharacterBackground.MUTANT_OUTCAST, CharacterBackground.RAIDER,
            CharacterBackground.SCAVENGER, CharacterBackground.SETTLEMENT_LEADER,
            CharacterBackground.TRIBAL_SHAMAN, CharacterBackground.VAULT_DWELLER,
            CharacterBackground.WASTELAND_DOCTOR, CharacterBackground.WASTELAND_SCOUT
        ],
        TTRPGGenre.COSMIC_HORROR: [
            CharacterBackground.ANTIQUARIAN, CharacterBackground.ASYLUM_PATIENT,
            CharacterBackground.CULT_SURVIVOR, CharacterBackground.CURSED_BLOODLINE,
            CharacterBackground.DREAM_TOUCHED, CharacterBackground.OCCULT_INVESTIGATOR,
            CharacterBackground.PROFESSOR, CharacterBackground.PSYCHIC_SENSITIVE
        ],
        TTRPGGenre.WESTERN: [
            CharacterBackground.BOUNTY_KILLER, CharacterBackground.CATTLE_RUSTLER,
            CharacterBackground.FRONTIER_DOCTOR, CharacterBackground.GUNSLINGER,
            CharacterBackground.HOMESTEADER, CharacterBackground.LAWMAN,
            CharacterBackground.OUTLAW, CharacterBackground.PREACHER,
            CharacterBackground.PROSPECTOR, CharacterBackground.RANCH_HAND,
            CharacterBackground.SALOON_OWNER, CharacterBackground.STAGE_DRIVER
        ],
        TTRPGGenre.SUPERHERO: [
            CharacterBackground.ALIEN_REFUGEE, CharacterBackground.GOVERNMENT_AGENT,
            CharacterBackground.LAB_ACCIDENT_SURVIVOR, CharacterBackground.MASKED_VIGILANTE,
            CharacterBackground.MILITARY_EXPERIMENT, CharacterBackground.MUTANT_ACTIVIST,
            CharacterBackground.REPORTER, CharacterBackground.SCIENTIST,
            CharacterBackground.SIDEKICK, CharacterBackground.TECH_GENIUS
        ]
    }
    
    @classmethod
    def get_random_motivation(cls) -> CharacterMotivation:
        """Get a random character motivation from extracted content."""
        return random.choice(list(CharacterMotivation))
    
    @classmethod
    def get_random_background(cls, genre: Optional[TTRPGGenre] = None) -> CharacterBackground:
        """Get a random background appropriate for the genre."""
        if genre:
            backgrounds = cls._GENRE_BACKGROUNDS.get(genre, list(CharacterBackground))
            return random.choice(backgrounds)
        
        return random.choice(list(CharacterBackground))

    # Story templates by background type
    ORIGIN_TEMPLATES = {
        "noble": [
            "Born into the {adjective} House of {family_name}, {name} was raised with privilege but yearned for {desire}.",
            "As the {order} child of Lord {family_name}, {name} was destined for {original_path} until {event} changed everything.",
            "The {family_name} estate was {name}'s prison and paradise, teaching lessons in both {virtue} and {vice}.",
        ],
        "commoner": [
            "{name} grew up in {location}, where life was {adjective} but {quality}.",
            "The child of {profession}s, {name} learned early that {lesson}.",
            "In the {adjective} streets of {location}, {name} discovered their talent for {skill}.",
        ],
        "outsider": [
            "Found as a child {location}, {name} never truly belonged anywhere.",
            "{name} arrived in {location} with no memory of {past_element}.",
            "Cast out from {origin} for {reason}, {name} wandered until finding {discovery}.",
        ],
        "tragedy": [
            "When {event} destroyed {precious_thing}, {name} swore to {oath}.",
            "{name} survived {disaster}, but the scars run deeper than flesh.",
            "The {adjective} night when {tragedy} occurred haunts {name} still.",
        ],
        # Genre-specific templates
        "cyberpunk": [
            "{name} jacked in for the first time at {age}, discovering a talent for {skill} that would define their life.",
            "Born in the {adjective} sprawl of {location}, {name} learned that {lesson} in the neon-lit streets.",
            "After {event} cost them their {precious_thing}, {name} replaced flesh with chrome, seeking {desire}.",
        ],
        "sci-fi": [
            "{name} was born on {location}, where {adjective} conditions forged their {quality} nature.",
            "As a {order} generation colonist, {name} inherited both {virtue} and {burden} from the stars.",
            "When {event} struck their {precious_thing}, {name} took to the void seeking {desire}.",
        ],
        "cosmic_horror": [
            "{name}'s research into {forbidden_topic} began innocently, but {event} revealed truths better left unknown.",
            "The {adjective} dreams started when {name} was {age}, whispers of {entity} echoing through their mind.",
            "After discovering {artifact} in {location}, {name} could never again see the world as others do.",
        ],
        "post_apocalyptic": [
            "{name} emerged from {location} into a world transformed by {disaster}, carrying only {precious_thing}.",
            "Born {time} after the {event}, {name} knows only the {adjective} wasteland and the law of {principle}.",
            "When {threat} destroyed their {community}, {name} learned that {lesson} in the ashes.",
        ],
        "western": [
            "{name} rode into {location} with {precious_thing} and a {adjective} reputation trailing behind.",
            "After {event} at {location}, {name} had no choice but to {action} and head for the frontier.",
            "The {adjective} dust of {location} couldn't hide {name}'s past, where {secret} waited to be revealed.",
        ],
        "superhero": [
            "The accident that gave {name} their powers also took their {precious_thing}, leaving them with {burden}.",
            "{name} discovered their abilities during {event}, realizing that with great power comes {responsibility}.",
            "Born with {quality}, {name} struggled to hide their true nature until {event} forced them to {action}.",
        ],
    }

    # Motivation templates by class archetype
    MOTIVATION_BY_ARCHETYPE = {
        "warrior": [
            "to prove their worth in battle",
            "to protect those who cannot protect themselves",
            "to find an honorable death",
            "to master the art of war",
        ],
        "mage": [
            "to unlock the secrets of the universe",
            "to master forbidden knowledge",
            "to prove that magic is the supreme power",
            "to find the source of their mysterious powers",
        ],
        "rogue": [
            "to pull off the ultimate heist",
            "to clear their name of false accusations",
            "to find the truth behind a conspiracy",
            "to survive by any means necessary",
        ],
        "divine": [
            "to serve their deity's will",
            "to spread their faith to the unenlightened",
            "to atone for past sins",
            "to understand the true nature of divinity",
        ],
        "nature": [
            "to protect the natural world",
            "to restore balance to the land",
            "to commune with primal forces",
            "to find their place in the natural order",
        ],
    }

    # Personality trait pools
    TRAIT_POOLS = {
        "brave": [
            "never backs down from a challenge",
            "faces danger with a smile",
            "inspires courage in others",
            "believes fear is a choice",
        ],
        "cunning": [
            "always has a backup plan",
            "sees angles others miss",
            "turns disadvantages into opportunities",
            "never reveals their true intentions",
        ],
        "wise": [
            "seeks understanding before action",
            "learns from every experience",
            "sees patterns others miss",
            "values knowledge above gold",
        ],
        "charismatic": [
            "can talk their way out of anything",
            "makes friends wherever they go",
            "inspires loyalty in others",
            "has a magnetic personality",
        ],
        "mysterious": [
            "keeps their past hidden",
            "speaks in riddles and metaphors",
            "appears when least expected",
            "knows more than they reveal",
        ],
    }

    # Cultural elements by race
    CULTURAL_ELEMENTS = {
        CharacterRace.ELF: {
            "values": ["grace", "nature", "artistry", "longevity"],
            "traditions": ["moonlit ceremonies", "tree singing", "star reading"],
            "conflicts": ["mortality of other races", "destruction of nature", "hasty decisions"],
        },
        CharacterRace.DWARF: {
            "values": ["honor", "craftsmanship", "tradition", "clan"],
            "traditions": ["forge rituals", "ancestor veneration", "stone carving"],
            "conflicts": ["dishonor", "shoddy work", "broken oaths"],
        },
        CharacterRace.HUMAN: {
            "values": ["ambition", "adaptability", "innovation", "legacy"],
            "traditions": ["coming of age", "harvest festivals", "guild ceremonies"],
            "conflicts": ["short lives", "internal strife", "rapid change"],
        },
        CharacterRace.HALFLING: {
            "values": ["comfort", "community", "simplicity", "joy"],
            "traditions": ["feast days", "gift giving", "storytelling"],
            "conflicts": ["adventure calling", "big folk problems", "leaving home"],
        },
        CharacterRace.TIEFLING: {
            "values": ["independence", "strength", "defiance", "loyalty"],
            "traditions": ["pact ceremonies", "blood bonds", "name earning"],
            "conflicts": ["prejudice", "infernal heritage", "trust issues"],
        },
        # Sci-Fi Races
        CharacterRace.CYBORG: {
            "values": ["efficiency", "enhancement", "transcendence", "logic"],
            "traditions": ["upgrade ceremonies", "data sharing", "system optimization"],
            "conflicts": ["humanity loss", "obsolescence", "maintenance costs"],
        },
        CharacterRace.ANDROID: {
            "values": ["purpose", "perfection", "service", "evolution"],
            "traditions": ["activation day", "core updates", "memory backups"],
            "conflicts": ["free will", "emotions", "identity"],
        },
        # Cyberpunk Races
        CharacterRace.AUGMENTED_HUMAN: {
            "values": ["power", "style", "edge", "survival"],
            "traditions": ["chrome rituals", "street reputation", "gang loyalty"],
            "conflicts": ["cyberpsychosis", "debt", "corporate control"],
        },
        # Post-Apocalyptic Races
        CharacterRace.MUTANT: {
            "values": ["adaptation", "survival", "community", "evolution"],
            "traditions": ["mutation rites", "rad-storm sheltering", "scav sharing"],
            "conflicts": ["pure strain prejudice", "instability", "rejection"],
        },
    }

    # Relationship templates
    RELATIONSHIPS = {
        "family": [
            {"type": "parent", "status": ["living", "deceased", "missing", "estranged"]},
            {"type": "sibling", "status": ["close", "rival", "lost", "unknown"]},
            {"type": "child", "status": ["hidden", "protected", "lost", "grown"]},
            {"type": "spouse", "status": ["devoted", "separated", "deceased", "searching for"]},
        ],
        "mentor": [
            {
                "type": "teacher",
                "relationship": ["grateful to", "betrayed by", "seeking", "surpassed"],
            },
            {
                "type": "master",
                "relationship": ["apprenticed to", "escaped from", "avenging", "honoring"],
            },
        ],
        "rival": [
            {
                "type": "competitor",
                "relationship": [
                    "friendly rivalry",
                    "bitter enemy",
                    "grudging respect",
                    "obsessed with",
                ],
            },
            {
                "type": "nemesis",
                "relationship": ["hunted by", "hunting", "locked in conflict", "destined to face"],
            },
        ],
        "ally": [
            {"type": "friend", "bond": ["childhood", "battle-forged", "oath-bound", "unlikely"]},
            {"type": "companion", "bond": ["loyal", "mysterious", "temporary", "supernatural"]},
        ],
    }

    def __init__(self, personality_manager=None, flavor_sources=None):
        """
        Initialize the backstory generator.

        Args:
            personality_manager: Optional personality manager for style
            flavor_sources: Optional flavor sources for narrative elements
        """
        self.personality_manager = personality_manager
        self.flavor_sources = flavor_sources or []

    def generate_backstory(
        self,
        character: Character,
        hints: Optional[str] = None,
        depth: str = "standard",  # "simple", "standard", "detailed"
        use_flavor_sources: bool = True,
    ) -> Backstory:
        """
        Generate a complete backstory for a character using enriched content.

        Args:
            character: The character to generate backstory for
            hints: Optional hints to guide generation
            depth: Level of detail for the backstory
            use_flavor_sources: Whether to incorporate flavor sources

        Returns:
            Complete Backstory object with story hooks and world connections
        """
        logger.info(f"Generating {depth} backstory for {character.name}")

        backstory = Backstory()

        # Determine narrative style based on system personality
        if self.personality_manager:
            style = self._get_narrative_style(character.system)
            backstory.narrative_style = style

        # Get enriched background from extracted content
        genre = getattr(character, 'genre', TTRPGGenre.FANTASY)
        enriched_background = self.get_random_background(genre)
        
        # Generate origin story with enriched content
        backstory.origin = self._generate_origin(character, hints)

        # Use enriched motivations from extracted content
        primary_motivation = self.get_random_motivation()
        backstory.motivation = self._format_motivation(primary_motivation, character)

        # Generate personality traits using enriched traits
        if depth == "detailed":
            character_traits = self.get_random_traits(count=5)
        elif depth == "standard":
            character_traits = self.get_random_traits(count=3)
        else:
            character_traits = self.get_random_traits(count=2)
        backstory.personality_traits = [trait.value.replace('_', ' ').title() for trait in character_traits]

        # Generate ideals, bonds, and flaws
        backstory.ideals = self._generate_ideals(character)
        backstory.bonds = self._generate_bonds(character)
        backstory.flaws = self._generate_flaws(character)

        # Generate goals and fears based on enriched motivations
        backstory.goals = self._generate_goals_from_motivations(character, depth)
        backstory.fears = self._generate_fears_from_motivations(character)

        # Generate relationships
        if depth in ["standard", "detailed"]:
            backstory.relationships = self._generate_relationships(character, depth)

        # Add story hooks for detailed backstories
        if depth == "detailed":
            backstory.story_hooks = self._generate_story_hooks(character, primary_motivation)
            backstory.world_connections = self._generate_world_connections(character, genre)

        # Add cultural references
        if character.race:
            backstory.cultural_references = self._get_cultural_references(character.race)

        # Incorporate flavor sources if available
        if use_flavor_sources and self.flavor_sources:
            self._incorporate_flavor_sources(backstory, character)

        # Generate background description with enriched background
        backstory.background = f"{enriched_background.value.replace('_', ' ').title()}: {self._generate_background_description(character, backstory, depth)}"

        logger.info(f"Backstory generation complete for {character.name}")
        return backstory

    def _get_narrative_style(self, system: str) -> str:
        """Get narrative style based on system personality."""
        if not self.personality_manager:
            return "neutral"

        # This would interface with the personality manager
        # For now, return style based on system
        styles = {
            "D&D 5e": "epic fantasy",
            "Call of Cthulhu": "cosmic horror",
            "Blades in the Dark": "Victorian crime",
            "Delta Green": "conspiracy thriller",
            "Pathfinder": "high fantasy",
            "Cyberpunk": "noir dystopian",
            "Traveller": "space opera",
            "Apocalypse World": "gritty survival",
            "Masks": "coming-of-age superhero",
            "Deadlands": "weird western",
        }

        return styles.get(system, "neutral")

    def _generate_origin(self, character: Character, hints: Optional[str]) -> str:
        """Generate character origin story."""
        # Check genre for genre-specific templates
        genre = getattr(character, 'genre', TTRPGGenre.FANTASY)
        
        # Map genres to template keys
        genre_template_map = {
            TTRPGGenre.CYBERPUNK: "cyberpunk",
            TTRPGGenre.SCI_FI: "sci-fi",
            TTRPGGenre.COSMIC_HORROR: "cosmic_horror",
            TTRPGGenre.POST_APOCALYPTIC: "post_apocalyptic",
            TTRPGGenre.WESTERN: "western",
            TTRPGGenre.SUPERHERO: "superhero",
        }
        
        # Determine origin type
        if genre in genre_template_map:
            origin_type = genre_template_map[genre]
        else:
            # Fantasy and other genres use traditional templates
            origin_types = ["noble", "commoner", "outsider", "tragedy"]
            
            if hints:
                # Parse hints for origin preferences
                if any(word in hints.lower() for word in ["noble", "royal", "lord"]):
                    origin_type = "noble"
                elif any(word in hints.lower() for word in ["tragic", "loss", "revenge"]):
                    origin_type = "tragedy"
                elif any(word in hints.lower() for word in ["mysterious", "unknown", "found"]):
                    origin_type = "outsider"
                else:
                    origin_type = "commoner"
            else:
                origin_type = random.choice(origin_types)

        # Get template and fill it
        templates = self.ORIGIN_TEMPLATES.get(origin_type, self.ORIGIN_TEMPLATES["commoner"])
        template = random.choice(templates)

        # Generate template variables with genre-specific options
        if genre == TTRPGGenre.CYBERPUNK:
            variables = self._generate_cyberpunk_variables(character)
        elif genre == TTRPGGenre.SCI_FI:
            variables = self._generate_scifi_variables(character)
        elif genre == TTRPGGenre.COSMIC_HORROR:
            variables = self._generate_cosmic_horror_variables(character)
        elif genre == TTRPGGenre.POST_APOCALYPTIC:
            variables = self._generate_post_apocalyptic_variables(character)
        elif genre == TTRPGGenre.WESTERN:
            variables = self._generate_western_variables(character)
        elif genre == TTRPGGenre.SUPERHERO:
            variables = self._generate_superhero_variables(character)
        else:
            # Default fantasy variables
            variables = {
                "name": character.name,
                "adjective": random.choice(["ancient", "renowned", "forgotten", "modest"]),
                "family_name": self._generate_family_name(character.race),
                "desire": random.choice(["adventure", "knowledge", "freedom", "purpose"]),
                "order": random.choice(["first", "second", "third", "youngest", "eldest"]),
                "original_path": random.choice(["politics", "military", "priesthood", "scholarship"]),
                "event": random.choice(
                    ["a prophetic dream", "a chance encounter", "a family tragedy", "a divine vision"]
                ),
                "location": self._generate_location(character.race),
                "quality": random.choice(["honest", "harsh", "simple", "dangerous"]),
                "profession": random.choice(["farmer", "merchant", "blacksmith", "innkeeper"]),
                "lesson": random.choice(
                    ["hard work pays off", "trust no one", "kindness matters", "strength prevails"]
                ),
                "skill": random.choice(["combat", "magic", "thievery", "diplomacy"]),
                "past_element": random.choice(
                    ["their past", "their true name", "their homeland", "their family"]
                ),
                "origin": self._generate_location(character.race),
                "reason": random.choice(
                    ["heresy", "a crime they didn't commit", "forbidden love", "speaking truth"]
                ),
                "discovery": random.choice(
                    ["a new purpose", "unexpected allies", "hidden strength", "their destiny"]
                ),
                "precious_thing": random.choice(
                    ["their home", "their family", "their innocence", "everything"]
                ),
                "oath": random.choice(
                    ["seek vengeance", "protect others", "find justice", "prevent it happening again"]
                ),
                "disaster": random.choice(["the great fire", "the plague", "the war", "the massacre"]),
                "tragedy": random.choice(["the betrayal", "the attack", "the ritual", "the accident"]),
                "vice": random.choice(["cruelty", "greed", "pride", "wrath"]),
            }

        # Format template with variables
        origin = template.format(**variables)

        return origin

    def _generate_motivation(self, character: Character) -> str:
        """Generate character motivation based on class."""
        # Map class to archetype
        class_archetypes = {
            CharacterClass.FIGHTER: "warrior",
            CharacterClass.BARBARIAN: "warrior",
            CharacterClass.PALADIN: "divine",
            CharacterClass.WIZARD: "mage",
            CharacterClass.SORCERER: "mage",
            CharacterClass.WARLOCK: "mage",
            CharacterClass.CLERIC: "divine",
            CharacterClass.DRUID: "nature",
            CharacterClass.RANGER: "nature",
            CharacterClass.ROGUE: "rogue",
            CharacterClass.MONK: "warrior",
            CharacterClass.BARD: "rogue",
            CharacterClass.ARTIFICER: "mage",
        }

        archetype = class_archetypes.get(character.character_class, "warrior")
        motivations = self.MOTIVATION_BY_ARCHETYPE[archetype]

        base_motivation = random.choice(motivations)

        # Add personal touch
        personal_element = random.choice(
            [
                f", driven by memories of {random.choice(['a lost love', 'a fallen comrade', 'a broken oath', 'a childhood dream'])}",
                f", seeking {random.choice(['redemption', 'answers', 'power', 'peace'])}",
                f", haunted by {random.choice(['the past', 'visions', 'a curse', 'guilt'])}",
                "",
            ]
        )

        return f"{character.name} adventures {base_motivation}{personal_element}"

    def _generate_personality_traits(self, character: Character, depth: str) -> List[str]:
        """Generate personality traits."""
        traits = []

        # Determine trait categories based on class
        class_traits = {
            CharacterClass.FIGHTER: ["brave", "disciplined", "protective"],
            CharacterClass.WIZARD: ["wise", "curious", "methodical"],
            CharacterClass.ROGUE: ["cunning", "adaptable", "independent"],
            CharacterClass.CLERIC: ["devoted", "compassionate", "righteous"],
            CharacterClass.BARBARIAN: ["fierce", "primal", "straightforward"],
            CharacterClass.PALADIN: ["noble", "just", "unwavering"],
            CharacterClass.SORCERER: ["passionate", "unpredictable", "confident"],
            CharacterClass.WARLOCK: ["mysterious", "ambitious", "complex"],
            CharacterClass.DRUID: ["wise", "natural", "balanced"],
            CharacterClass.RANGER: ["observant", "self-reliant", "practical"],
            CharacterClass.MONK: ["disciplined", "serene", "focused"],
            CharacterClass.BARD: ["charismatic", "creative", "versatile"],
            CharacterClass.ARTIFICER: ["inventive", "analytical", "precise"],
        }

        trait_categories = class_traits.get(character.character_class, ["brave", "cunning", "wise"])

        # Number of traits based on depth
        num_traits = {"simple": 1, "standard": 2, "detailed": 3}.get(depth, 2)

        for i in range(num_traits):
            category = random.choice(trait_categories)
            if category in self.TRAIT_POOLS:
                trait_base = random.choice(self.TRAIT_POOLS[category])
                traits.append(f"{character.name} {trait_base}")
            else:
                # Generic trait
                traits.append(self._generate_generic_trait(character))

        return traits

    def _generate_generic_trait(self, character: Character) -> str:
        """Generate a generic personality trait."""
        templates = [
            "I judge people by their actions, not their words.",
            "If someone is in trouble, I'm always ready to lend help.",
            "I'm confident in my own abilities and do what I can to instill confidence in others.",
            "I don't like to get my hands dirty unless I have to.",
            "I'm always polite and respectful.",
            "I'm haunted by memories of war.",
            "I've lost too many friends, and I'm slow to make new ones.",
            "I have a crude sense of humor.",
            "I face problems head-on.",
            "I enjoy being strong and like breaking things.",
        ]

        return random.choice(templates)

    def _generate_ideals(self, character: Character) -> List[str]:
        """Generate character ideals."""
        alignment_ideals = {
            "Good": [
                "Greater Good. My gifts are meant to be shared with all.",
                "Respect. People deserve to be treated with dignity.",
                "Protection. I must protect those who cannot protect themselves.",
            ],
            "Evil": [
                "Power. Knowledge is the path to power and domination.",
                "Might. The strong are meant to rule over the weak.",
                "Greed. I will do whatever it takes to become wealthy.",
            ],
            "Lawful": [
                "Tradition. Ancient traditions must be preserved.",
                "Honor. My word is my bond.",
                "Responsibility. I do what I must and obey just authority.",
            ],
            "Chaotic": [
                "Freedom. Chains are meant to be broken.",
                "Change. We must help bring about change.",
                "Independence. I must prove I can handle myself.",
            ],
            "Neutral": [
                "Balance. There must be balance in all things.",
                "Knowledge. Understanding is more important than faith.",
                "Self-Improvement. I must constantly improve myself.",
            ],
        }

        # Parse alignment
        if "Good" in character.alignment:
            ideal_pool = alignment_ideals["Good"]
        elif "Evil" in character.alignment:
            ideal_pool = alignment_ideals["Evil"]
        elif "Lawful" in character.alignment:
            ideal_pool = alignment_ideals["Lawful"]
        elif "Chaotic" in character.alignment:
            ideal_pool = alignment_ideals["Chaotic"]
        else:
            ideal_pool = alignment_ideals["Neutral"]

        return [random.choice(ideal_pool)]

    def _generate_bonds(self, character: Character) -> List[str]:
        """Generate character bonds."""
        bond_templates = [
            "I would die to recover an ancient relic of my faith.",
            "I will someday get revenge on the corrupt hierarchy.",
            "I owe my life to the priest who took me in when my parents died.",
            "Everything I do is for the common people.",
            "I will do anything to protect the temple where I served.",
            "I seek to preserve a sacred text that my enemies seek to destroy.",
            "My family, clan, or tribe is the most important thing in my life.",
            "An injury to the unspoiled wilderness is an injury to me.",
            "I am the last of my tribe, and it is up to me to see their names remembered.",
            "I suffer awful visions of a coming disaster.",
        ]

        return [random.choice(bond_templates)]

    def _generate_flaws(self, character: Character) -> List[str]:
        """Generate character flaws."""
        flaw_templates = [
            "I judge others harshly, and myself even more severely.",
            "I put too much trust in those who wield power.",
            "My piety sometimes leads me to blindly trust divine authority.",
            "I am inflexible in my thinking.",
            "I am suspicious of strangers and expect the worst of them.",
            "Once I pick a goal, I become obsessed with it.",
            "I can't resist a pretty face.",
            "I'm always in debt from my expensive tastes.",
            "I'm convinced that no one could ever fool me.",
            "I'm too greedy for my own good.",
        ]

        return [random.choice(flaw_templates)]

    def _generate_goals(self, character: Character, depth: str) -> List[str]:
        """Generate character goals."""
        num_goals = {"simple": 1, "standard": 2, "detailed": 3}.get(depth, 2)

        goal_templates = [
            "Master the ancient techniques of my order",
            "Find the artifact that was stolen from my family",
            "Prove myself worthy of my heritage",
            "Uncover the truth about my mysterious past",
            "Build a lasting legacy",
            "Protect the innocent from evil",
            "Gain enough power to never be helpless again",
            "Find a place where I truly belong",
            "Redeem myself for past failures",
            "Unlock the secrets of ultimate power",
        ]

        goals = []
        for _ in range(num_goals):
            goals.append(random.choice(goal_templates))

        return goals

    def _generate_fears(self, character: Character) -> List[str]:
        """Generate character fears."""
        fear_templates = [
            "Losing control of my power",
            "Being abandoned by those I care about",
            "Failing when others depend on me",
            "The darkness within myself",
            "Being powerless to help",
            "My past catching up with me",
            "Dying before achieving my goals",
            "Being forgotten",
            "Confined spaces",
            "The unknown",
        ]

        return [random.choice(fear_templates)]

    def _generate_relationships(self, character: Character, depth: str) -> List[Dict[str, str]]:
        """Generate character relationships."""
        relationships = []

        num_relationships = {"simple": 1, "standard": 2, "detailed": 4}.get(depth, 2)

        for _ in range(num_relationships):
            category = random.choice(list(self.RELATIONSHIPS.keys()))
            relationship_type = random.choice(self.RELATIONSHIPS[category])

            # Determine relationship status clearly
            if "status" in relationship_type:
                status = random.choice(relationship_type.get("status", ["unknown"]))
            elif "relationship" in relationship_type:
                status = relationship_type.get("relationship", ["connected"])[0]
            else:
                status = relationship_type.get("bond", ["connected"])[0]

            rel = {
                "category": category,
                "type": relationship_type["type"],
                "name": self._generate_npc_name(),
                "status": status,
                "description": self._generate_relationship_description(category, relationship_type),
            }

            relationships.append(rel)

        return relationships

    def _generate_npc_name(self) -> str:
        """Generate a random NPC name."""
        first_names = ["Marcus", "Elena", "Theron", "Lyra", "Gareth", "Mira"]
        last_names = ["Blackwood", "Stormwind", "Ironforge", "Moonwhisper", "Shadowbane"]

        return f"{random.choice(first_names)} {random.choice(last_names)}"

    def _generate_relationship_description(
        self, category: str, relationship_type: Dict[str, Any]
    ) -> str:
        """Generate a description for a relationship."""
        descriptions = {
            "family": "Blood ties that bind, for better or worse.",
            "mentor": "One who shaped the path I now walk.",
            "rival": "Competition drives us both to greatness.",
            "ally": "A bond forged through shared trials.",
        }

        return descriptions.get(category, "An important person in my life.")

    def _get_cultural_references(self, race: CharacterRace) -> List[str]:
        """Get cultural references for a race."""
        if race in self.CULTURAL_ELEMENTS:
            elements = self.CULTURAL_ELEMENTS[race]
            references = []

            # Add values
            if "values" in elements:
                references.append(f"Values: {', '.join(elements['values'])}")

            # Add traditions
            if "traditions" in elements:
                references.append(
                    f"Traditions: {', '.join(random.sample(elements['traditions'], min(2, len(elements['traditions']))))}"
                )

            # Add conflicts
            if "conflicts" in elements:
                references.append(f"Struggles with: {random.choice(elements['conflicts'])}")

            return references

        return []

    def _incorporate_flavor_sources(self, backstory: Backstory, character: Character):
        """Incorporate elements from flavor sources."""
        # This would interface with flavor source documents
        # For now, just add a note
        if self.flavor_sources:
            backstory.cultural_references.append("Elements drawn from campaign setting lore")

    def _generate_background_description(
        self, character: Character, backstory: Backstory, depth: str
    ) -> str:
        """Generate a cohesive background description."""
        description_parts = []

        # Add origin
        if backstory.origin:
            description_parts.append(backstory.origin)

        # Add motivation
        if backstory.motivation:
            description_parts.append(backstory.motivation)

        # Add personality summary
        if backstory.personality_traits:
            trait_summary = f"{character.name} is known for: {backstory.personality_traits[0]}"
            description_parts.append(trait_summary)

        # Add relationships if detailed
        if depth == "detailed" and backstory.relationships:
            rel = backstory.relationships[0]
            rel_summary = f"Their {rel['category']}, {rel['name']}, {rel['description'].lower()}"
            description_parts.append(rel_summary)

        # Add goals
        if backstory.goals:
            goal_summary = f"Now, {character.name} seeks to {backstory.goals[0].lower()}"
            description_parts.append(goal_summary)

        return " ".join(description_parts)

    def _generate_family_name(self, race: Optional[CharacterRace]) -> str:
        """Generate a family name based on race."""
        family_names = {
            CharacterRace.HUMAN: ["Winters", "Blackstone", "Goldshire", "Brightblade"],
            CharacterRace.ELF: ["Moonwhisper", "Starweaver", "Silverleaf", "Windrunner"],
            CharacterRace.DWARF: ["Ironforge", "Stonebeard", "Battlehammer", "Goldbeard"],
            CharacterRace.HALFLING: ["Goodbarrel", "Greenhill", "Proudfoot", "Tosscobble"],
            None: ["Smith", "Walker", "Hunter", "Miller"],
        }

        names = family_names.get(race, family_names[None])
        return random.choice(names)

    def _generate_location(self, race: Optional[CharacterRace]) -> str:
        """Generate a location name based on race."""
        # Check if character has a genre
        genre = TTRPGGenre.FANTASY  # Default
        if hasattr(self, '_current_character') and hasattr(self._current_character, 'genre'):
            genre = self._current_character.genre
        
        # Genre-specific locations
        if genre == TTRPGGenre.CYBERPUNK:
            return random.choice(["Night City", "Neo-Tokyo", "The Sprawl", "Chrome District"])
        elif genre == TTRPGGenre.SCI_FI:
            return random.choice(["Mars Colony", "Titan Station", "Alpha Centauri", "The Belt"])
        elif genre == TTRPGGenre.COSMIC_HORROR:
            return random.choice(["Arkham", "Innsmouth", "Dunwich", "Miskatonic University"])
        elif genre == TTRPGGenre.POST_APOCALYPTIC:
            return random.choice(["The Wasteland", "Vault 13", "New Vegas", "The Citadel"])
        elif genre == TTRPGGenre.WESTERN:
            return random.choice(["Tombstone", "Deadwood", "Dodge City", "El Paso"])
        elif genre == TTRPGGenre.SUPERHERO:
            return random.choice(["Metropolis", "Gotham City", "Central City", "New York"])
        else:
            # Fantasy locations
            locations = {
                CharacterRace.HUMAN: ["Waterdeep", "Baldur's Gate", "Neverwinter", "King's Landing"],
                CharacterRace.ELF: ["Silverymoon", "Evermeet", "Myth Drannor", "the Feywild"],
                CharacterRace.DWARF: ["Mithral Hall", "Ironforge", "the Lonely Mountain", "Gauntlgrym"],
                CharacterRace.HALFLING: ["the Shire", "Luiren", "Green Fields", "Hobbiton"],
                None: ["a distant land", "the frontier", "the old kingdom", "parts unknown"],
            }
            locs = locations.get(race, locations[None])
            return random.choice(locs)

    def _generate_cyberpunk_variables(self, character: Character) -> Dict[str, Any]:
        """Generate Cyberpunk-specific template variables."""
        return {
            "name": character.name,
            "age": random.choice(["twelve", "fifteen", "eighteen", "twenty-one"]),
            "adjective": random.choice(["neon-soaked", "chrome-plated", "data-corrupted", "augmented"]),
            "location": self._generate_location(character.race),
            "skill": random.choice(["netrunning", "combat", "hacking", "dealing", "fixing"]),
            "lesson": random.choice(["chrome is king", "data is power", "trust no corp", "meat is weak"]),
            "event": random.choice(["a botched run", "corporate betrayal", "a gang war", "cyberpsychosis"]),
            "precious_thing": random.choice(["humanity", "memories", "organic body", "freedom"]),
            "desire": random.choice(["revenge", "freedom", "power", "transcendence"]),
            "family_name": random.choice(["Chrome", "Neon", "Binary", "Ghost"]),
            "quality": random.choice(["ruthless", "street-smart", "augmented", "connected"]),
        }

    def _generate_scifi_variables(self, character: Character) -> Dict[str, Any]:
        """Generate Sci-Fi-specific template variables."""
        return {
            "name": character.name,
            "adjective": random.choice(["stellar", "void-touched", "gravity-born", "synthetic"]),
            "location": self._generate_location(character.race),
            "order": random.choice(["first", "third", "fifth", "tenth"]),
            "virtue": random.choice(["exploration", "discovery", "unity", "progress"]),
            "burden": random.choice(["isolation", "responsibility", "alien heritage", "time debt"]),
            "event": random.choice(["solar flare", "alien contact", "jump failure", "colony collapse"]),
            "precious_thing": random.choice(["homeworld", "ship", "crew", "memories"]),
            "desire": random.choice(["home", "discovery", "peace", "understanding"]),
            "quality": random.choice(["adaptable", "resilient", "curious", "determined"]),
        }

    def _generate_cosmic_horror_variables(self, character: Character) -> Dict[str, Any]:
        """Generate Cosmic Horror-specific template variables."""
        return {
            "name": character.name,
            "age": random.choice(["young", "middle-aged", "elderly", "uncertain"]),
            "adjective": random.choice(["unspeakable", "forbidden", "eldritch", "sanity-blasting"]),
            "location": self._generate_location(character.race),
            "forbidden_topic": random.choice(["ancient texts", "stellar alignments", "dreams", "genealogy"]),
            "event": random.choice(["the ritual", "the summoning", "the awakening", "the revelation"]),
            "entity": random.choice(["the Old Ones", "something ancient", "the void", "forgotten gods"]),
            "artifact": random.choice(["the tome", "the idol", "the medallion", "the map"]),
            "precious_thing": random.choice(["sanity", "innocence", "faith", "humanity"]),
        }

    def _generate_post_apocalyptic_variables(self, character: Character) -> Dict[str, Any]:
        """Generate Post-Apocalyptic-specific template variables."""
        return {
            "name": character.name,
            "adjective": random.choice(["irradiated", "desolate", "brutal", "scarred"]),
            "location": self._generate_location(character.race),
            "disaster": random.choice(["the bombs", "the plague", "the collapse", "the war"]),
            "time": random.choice(["generations", "decades", "years", "cycles"]),
            "event": random.choice(["the bombs", "the plague", "the collapse", "the great dying"]),
            "precious_thing": random.choice(["clean water", "medicine", "ammunition", "hope"]),
            "principle": random.choice(["survival", "strength", "scavenging", "mutation"]),
            "threat": random.choice(["raiders", "radiation", "mutants", "starvation"]),
            "community": random.choice(["settlement", "vault", "tribe", "caravan"]),
            "lesson": random.choice(["trust kills", "strength survives", "adapt or die", "hope is dangerous"]),
        }

    def _generate_western_variables(self, character: Character) -> Dict[str, Any]:
        """Generate Western-specific template variables."""
        return {
            "name": character.name,
            "adjective": random.choice(["dusty", "lawless", "frontier", "wild"]),
            "location": self._generate_location(character.race),
            "precious_thing": random.choice(["six-shooter", "horse", "badge", "gold"]),
            "event": random.choice(["the shootout", "the hanging", "the robbery", "the betrayal"]),
            "action": random.choice(["draw iron", "ride hard", "face justice", "seek revenge"]),
            "secret": random.choice(["a bounty", "a murder", "stolen gold", "true identity"]),
            "reputation": random.choice(["dangerous", "mysterious", "deadly", "honorable"]),
        }

    def _generate_superhero_variables(self, character: Character) -> Dict[str, Any]:
        """Generate Superhero-specific template variables."""
        return {
            "name": character.name,
            "precious_thing": random.choice(["normal life", "loved ones", "innocence", "humanity"]),
            "burden": random.choice(["responsibility", "guilt", "destiny", "power"]),
            "event": random.choice(["the accident", "the experiment", "the awakening", "the attack"]),
            "responsibility": random.choice(["great responsibility", "protecting others", "justice", "hope"]),
            "quality": random.choice(["extraordinary abilities", "great power", "unique gifts", "mutations"]),
            "action": random.choice(["reveal themselves", "become a hero", "embrace destiny", "fight back"]),
        }
    
    def _format_motivation(self, motivation: CharacterMotivation, character: Character) -> str:
        """Format a motivation enum into a narrative string."""
        motivation_descriptions = {
            CharacterMotivation.ADVENTURE: "seeks thrilling adventures and new experiences",
            CharacterMotivation.KNOWLEDGE: "hungers for knowledge and understanding",
            CharacterMotivation.POWER: "desires power and influence over others",
            CharacterMotivation.REDEMPTION: "seeks redemption for past mistakes",
            CharacterMotivation.REVENGE: "burns with desire for vengeance",
            CharacterMotivation.WEALTH: "pursues riches and material wealth",
            CharacterMotivation.HONOR: "strives to uphold honor and duty",
            CharacterMotivation.FREEDOM: "yearns for freedom and independence",
            CharacterMotivation.JUSTICE: "fights for justice and fairness",
            CharacterMotivation.LOVE: "searches for love and connection",
            CharacterMotivation.SURVIVAL: "struggles to survive against all odds",
            CharacterMotivation.LEGACY: "works to leave a lasting legacy",
            CharacterMotivation.PROTECTION: "dedicates themselves to protecting others",
            CharacterMotivation.DISCOVERY: "driven to discover the unknown",
            CharacterMotivation.ACCEPTANCE: "seeks acceptance and belonging",
        }
        
        base_description = motivation_descriptions.get(
            motivation, 
            f"driven by {motivation.value.replace('_', ' ')}"
        )
        
        return f"{character.name} {base_description}"
    
    def _generate_goals_from_motivations(self, character: Character, depth: str = "standard") -> List[str]:
        """Generate character goals based on enriched motivations."""
        num_goals = 3 if depth == "detailed" else 2 if depth == "standard" else 1
        motivations = random.sample(list(CharacterMotivation), min(num_goals, len(CharacterMotivation)))
        goals = []
        
        for motivation in motivations:
            if "FINDING" in motivation.name or "DISCOVERING" in motivation.name:
                goals.append(f"To {motivation.value.replace('_', ' ').lower()}")
            elif "PROTECTING" in motivation.name or "DEFENDING" in motivation.name:
                goals.append(f"To {motivation.value.replace('_', ' ').lower()}")
            elif motivation.name.endswith("ING"):
                goals.append(f"To continue {motivation.value.replace('_', ' ').lower()}")
            else:
                goals.append(f"To achieve {motivation.value.replace('_', ' ').lower()}")
        
        return goals
    
    def _generate_fears_from_motivations(self, character: Character) -> List[str]:
        """Generate character fears based on enriched content."""
        fear_motivations = [
            m for m in CharacterMotivation 
            if m.name in ['ABANDONMENT', 'BETRAYAL', 'CHAOS', 'DEATH', 'FAILURE',
                         'HELPLESSNESS', 'ISOLATION', 'LOSS', 'MADNESS', 'POWERLESSNESS']
        ]
        
        selected_fears = random.sample(fear_motivations, min(2, len(fear_motivations)))
        return [f"Fear of {fear.value.replace('_', ' ')}" for fear in selected_fears]
    
    def _generate_story_hooks(self, character: Character, motivation: CharacterMotivation) -> List[StoryHook]:
        """Generate story hooks for character integration."""
        hooks = []
        genre = getattr(character, 'genre', TTRPGGenre.FANTASY)
        
        # Quest hook based on motivation
        quest_hook = StoryHook(
            hook_type="quest",
            title=f"The {motivation.value.replace('_', ' ').title()} Quest",
            description=self._generate_quest_description(character, motivation),
            urgency=random.choice(["low", "medium", "high", "critical"]),
            stakes=self._generate_stakes(motivation),
            potential_allies=[self._generate_ally_name() for _ in range(random.randint(1, 3))],
            potential_enemies=[self._generate_enemy_name() for _ in range(random.randint(1, 2))],
            rewards=self._generate_rewards(genre),
            complications=self._generate_complications(genre),
            genre_tags=[genre.value if hasattr(genre, 'value') else str(genre)]
        )
        hooks.append(quest_hook)
        
        # Mystery hook
        if random.random() > 0.5:
            mystery_hook = StoryHook(
                hook_type="mystery",
                title=f"The {random.choice(['Hidden', 'Lost', 'Forbidden', 'Ancient'])} {random.choice(['Truth', 'Secret', 'Knowledge', 'Artifact'])}",
                description=self._generate_mystery_description(character),
                urgency="medium",
                stakes="Unknown consequences if the mystery remains unsolved",
                potential_allies=[self._generate_ally_name()],
                potential_enemies=[],
                rewards=["Knowledge", "Understanding"],
                complications=["Deceptive clues", "False leads"],
                genre_tags=[genre.value if hasattr(genre, 'value') else str(genre)]
            )
            hooks.append(mystery_hook)
        
        return hooks
    
    def _generate_world_connections(self, character: Character, genre: TTRPGGenre) -> List[WorldElement]:
        """Generate world-building elements connected to the character."""
        connections = []
        
        # Home location
        home = WorldElement(
            element_type="location",
            name=self._generate_location(character.race),
            description=f"The place where {character.name} spent their formative years",
            significance="Character's origin point",
            history="A settlement with its own stories and secrets",
            current_state=random.choice(["Thriving", "Struggling", "Abandoned", "Under threat"]),
            connections=[character.name],
            secrets=[f"Hidden {random.choice(['treasure', 'danger', 'knowledge', 'power'])}"],
            rumors=["Strange happenings at night", "Visitors from afar"],
            genre=genre.value if hasattr(genre, 'value') else None
        )
        connections.append(home)
        
        # Important faction
        if random.random() > 0.5:
            faction = WorldElement(
                element_type="faction",
                name=self._generate_faction_name(genre),
                description="An organization with influence in the region",
                significance="Potential ally or enemy",
                history="Founded generations ago",
                current_state=random.choice(["Rising power", "Declining influence", "Hidden agenda"]),
                connections=[character.name, "Various NPCs"],
                secrets=["True leadership", "Hidden goals"],
                rumors=["Recruiting members", "Planning something big"],
                genre=genre.value if hasattr(genre, 'value') else None
            )
            connections.append(faction)
        
        return connections
    
    def _generate_quest_description(self, character: Character, motivation: CharacterMotivation) -> str:
        """Generate a quest description based on character and motivation."""
        templates = [
            f"{character.name} must {motivation.value.replace('_', ' ').lower()} to fulfill their destiny",
            f"A quest that will test {character.name}'s resolve to {motivation.value.replace('_', ' ').lower()}",
            f"An opportunity for {character.name} to pursue their goal of {motivation.value.replace('_', ' ').lower()}"
        ]
        return random.choice(templates)
    
    def _generate_stakes(self, motivation: CharacterMotivation) -> str:
        """Generate stakes for a story hook."""
        stakes_map = {
            "SURVIVAL": "Life or death",
            "POWER": "Control of the region",
            "KNOWLEDGE": "Understanding of fundamental truths",
            "REDEMPTION": "Chance for forgiveness",
            "REVENGE": "Justice or eternal regret",
            "WEALTH": "Fortune or poverty",
            "HONOR": "Reputation and legacy",
            "FREEDOM": "Liberty or enslavement",
            "JUSTICE": "Right prevails or evil triumphs",
            "LOVE": "Together or forever apart"
        }
        
        for key, value in stakes_map.items():
            if key in motivation.name:
                return value
        
        return "Personal fulfillment or crushing failure"
    
    def _generate_rewards(self, genre: TTRPGGenre) -> List[str]:
        """Generate appropriate rewards for the genre."""
        base_rewards = ["Experience", "Reputation", "Allies"]
        
        genre_rewards = {
            TTRPGGenre.FANTASY: ["Magic items", "Gold", "Land", "Titles"],
            TTRPGGenre.SCI_FI: ["Technology", "Credits", "Ship upgrades", "Data"],
            TTRPGGenre.CYBERPUNK: ["Cyberware", "Information", "Street cred", "Nuyen"],
            TTRPGGenre.POST_APOCALYPTIC: ["Supplies", "Weapons", "Safe haven", "Clean water"],
            TTRPGGenre.COSMIC_HORROR: ["Forbidden knowledge", "Sanity", "Artifacts", "Truth"],
            TTRPGGenre.WESTERN: ["Gold", "Land deed", "Pardons", "Horses"],
            TTRPGGenre.SUPERHERO: ["Public acclaim", "Tech upgrades", "Team membership", "Information"]
        }
        
        rewards = base_rewards.copy()
        if genre in genre_rewards:
            rewards.extend(random.sample(genre_rewards[genre], min(2, len(genre_rewards[genre]))))
        
        return rewards
    
    def _generate_complications(self, genre: TTRPGGenre) -> List[str]:
        """Generate complications appropriate for the genre."""
        base_complications = ["Betrayal", "Time pressure", "Moral dilemma"]
        
        genre_complications = {
            TTRPGGenre.FANTASY: ["Ancient curse", "Divine intervention", "Dragon involvement"],
            TTRPGGenre.SCI_FI: ["System malfunction", "Alien interference", "Jump drive failure"],
            TTRPGGenre.CYBERPUNK: ["Corporate involvement", "Netrunner opposition", "Gang war"],
            TTRPGGenre.POST_APOCALYPTIC: ["Radiation zone", "Mutant horde", "Resource scarcity"],
            TTRPGGenre.COSMIC_HORROR: ["Sanity loss", "Cultist interference", "Reality distortion"],
            TTRPGGenre.WESTERN: ["Law enforcement", "Native tensions", "Desert conditions"],
            TTRPGGenre.SUPERHERO: ["Collateral damage", "Secret identity risk", "Villain team-up"]
        }
        
        complications = base_complications.copy()
        if genre in genre_complications:
            complications.extend(random.sample(genre_complications[genre], min(2, len(genre_complications[genre]))))
        
        return complications
    
    def _generate_ally_name(self) -> str:
        """Generate a random ally name."""
        first_names = ["Marcus", "Elena", "Theron", "Lyra", "Gareth", "Mira", "Dex", "Nova"]
        descriptors = ["the Wise", "the Bold", "Shadowstep", "Brightblade", "of the North", "the Scholar"]
        return f"{random.choice(first_names)} {random.choice(descriptors)}"
    
    def _generate_enemy_name(self) -> str:
        """Generate a random enemy name."""
        titles = ["Lord", "Lady", "Captain", "The", "Warlord", "Master"]
        names = ["Blackheart", "Vex", "Malthus", "Crimson", "Void", "Bane", "Ruin", "Shade"]
        return f"{random.choice(titles)} {random.choice(names)}"
    
    def _generate_mystery_description(self, character: Character) -> str:
        """Generate a mystery description."""
        mysteries = [
            f"Strange symbols appearing that only {character.name} can see",
            f"A message from {character.name}'s past that changes everything",
            f"An artifact that responds only to {character.name}",
            f"Visions plaguing {character.name} that seem to predict the future"
        ]
        return random.choice(mysteries)
    
    def _generate_faction_name(self, genre: TTRPGGenre) -> str:
        """Generate a faction name appropriate for the genre."""
        genre_factions = {
            TTRPGGenre.FANTASY: ["Order of the Silver Dawn", "Merchants' Guild", "Circle of Mages"],
            TTRPGGenre.SCI_FI: ["Colonial Authority", "Free Traders Union", "Science Directorate"],
            TTRPGGenre.CYBERPUNK: ["Arasaka Corp", "Street Samurai Clan", "Data Liberation Front"],
            TTRPGGenre.POST_APOCALYPTIC: ["New Republic", "Scavenger Coalition", "Vault-Tec Remnants"],
            TTRPGGenre.COSMIC_HORROR: ["Esoteric Order", "Department of Investigations", "The Watchers"],
            TTRPGGenre.WESTERN: ["Railroad Company", "Cattlemen's Association", "Pinkerton Agency"],
            TTRPGGenre.SUPERHERO: ["Hero League", "S.H.I.E.L.D.", "Villain Syndicate"]
        }
        
        factions = genre_factions.get(genre, ["The Guild", "The Order", "The Alliance"])
        return random.choice(factions)
