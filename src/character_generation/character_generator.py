"""Character generation engine for TTRPG Assistant."""

import logging
import random
from typing import List, Optional

from .models import (
    Backstory,
    Character,
    CharacterClass,
    CharacterRace,
    CharacterStats,
    Equipment,
    ExtendedCharacter,
    GenreSpecificData,
    TTRPGGenre,
    CyberpunkAugmentation,
    SciFiTechnology,
    CosmicHorrorSanity,
    PostApocalypticMutation,
    SuperheroPower,
)
from .validators import CharacterValidator, ValidationError

logger = logging.getLogger(__name__)


class CharacterGenerator:
    """Generate player characters for various TTRPG systems."""

    # D&D 5e stat arrays
    STANDARD_ARRAY = [15, 14, 13, 12, 10, 8]

    # Racial stat bonuses (D&D 5e style)
    RACIAL_BONUSES = {
        CharacterRace.HUMAN: {"all": 1},
        CharacterRace.ELF: {"dexterity": 2},
        CharacterRace.DWARF: {"constitution": 2},
        CharacterRace.HALFLING: {"dexterity": 2},
        CharacterRace.ORC: {"strength": 2, "constitution": 1},
        CharacterRace.TIEFLING: {"charisma": 2, "intelligence": 1},
        CharacterRace.DRAGONBORN: {"strength": 2, "charisma": 1},
        CharacterRace.GNOME: {"intelligence": 2},
        CharacterRace.HALF_ELF: {"charisma": 2, "any_two": 1},
        CharacterRace.HALF_ORC: {"strength": 2, "constitution": 1},
    }

    # Class primary stats (for stat array assignment)
    CLASS_PRIMARY_STATS = {
        CharacterClass.FIGHTER: ["strength", "constitution"],
        CharacterClass.WIZARD: ["intelligence", "constitution"],
        CharacterClass.CLERIC: ["wisdom", "constitution"],
        CharacterClass.ROGUE: ["dexterity", "intelligence"],
        CharacterClass.RANGER: ["dexterity", "wisdom"],
        CharacterClass.PALADIN: ["strength", "charisma"],
        CharacterClass.BARBARIAN: ["strength", "constitution"],
        CharacterClass.SORCERER: ["charisma", "constitution"],
        CharacterClass.WARLOCK: ["charisma", "constitution"],
        CharacterClass.DRUID: ["wisdom", "constitution"],
        CharacterClass.MONK: ["dexterity", "wisdom"],
        CharacterClass.BARD: ["charisma", "dexterity"],
        CharacterClass.ARTIFICER: ["intelligence", "constitution"],
    }

    # Starting equipment by class
    CLASS_EQUIPMENT = {
        CharacterClass.FIGHTER: {
            "weapons": ["Longsword", "Shield"],
            "armor": ["Chain Mail"],
            "items": ["Backpack", "Bedroll", "Rations (10 days)"],
        },
        CharacterClass.WIZARD: {
            "weapons": ["Quarterstaff"],
            "armor": [],
            "items": ["Spellbook", "Component Pouch", "Scholar's Pack"],
        },
        CharacterClass.ROGUE: {
            "weapons": ["Shortsword", "Dagger (2)"],
            "armor": ["Leather Armor"],
            "items": ["Thieves' Tools", "Burglar's Pack"],
        },
        CharacterClass.CLERIC: {
            "weapons": ["Mace"],
            "armor": ["Scale Mail", "Shield"],
            "items": ["Holy Symbol", "Priest's Pack"],
        },
        CharacterClass.RANGER: {
            "weapons": ["Longbow", "Shortsword (2)"],
            "armor": ["Leather Armor"],
            "items": ["Explorer's Pack", "Arrows (40)"],
        },
        CharacterClass.BARBARIAN: {
            "weapons": ["Greataxe", "Handaxe (2)"],
            "armor": [],
            "items": ["Explorer's Pack", "Javelins (4)"],
        },
        CharacterClass.PALADIN: {
            "weapons": ["Longsword", "Shield"],
            "armor": ["Chain Mail"],
            "items": ["Holy Symbol", "Priest's Pack"],
        },
        CharacterClass.SORCERER: {
            "weapons": ["Dagger (2)"],
            "armor": [],
            "items": ["Component Pouch", "Explorer's Pack"],
        },
        CharacterClass.WARLOCK: {
            "weapons": ["Light Crossbow", "Dagger (2)"],
            "armor": ["Leather Armor"],
            "items": ["Component Pouch", "Scholar's Pack"],
        },
        CharacterClass.DRUID: {
            "weapons": ["Scimitar"],
            "armor": ["Leather Armor", "Wooden Shield"],
            "items": ["Druidic Focus", "Explorer's Pack"],
        },
        CharacterClass.MONK: {
            "weapons": ["Shortsword"],
            "armor": [],
            "items": ["Explorer's Pack", "Darts (10)"],
        },
        CharacterClass.BARD: {
            "weapons": ["Rapier", "Dagger"],
            "armor": ["Leather Armor"],
            "items": ["Musical Instrument", "Entertainer's Pack"],
        },
        CharacterClass.ARTIFICER: {
            "weapons": ["Light Crossbow"],
            "armor": ["Scale Mail", "Shield"],
            "items": ["Thieves' Tools", "Dungeoneer's Pack"],
        },
    }

    # Hit dice by class
    HIT_DICE = {
        CharacterClass.FIGHTER: 10,
        CharacterClass.WIZARD: 6,
        CharacterClass.CLERIC: 8,
        CharacterClass.ROGUE: 8,
        CharacterClass.RANGER: 10,
        CharacterClass.PALADIN: 10,
        CharacterClass.BARBARIAN: 12,
        CharacterClass.SORCERER: 6,
        CharacterClass.WARLOCK: 8,
        CharacterClass.DRUID: 8,
        CharacterClass.MONK: 8,
        CharacterClass.BARD: 8,
        CharacterClass.ARTIFICER: 8,
    }

    def __init__(self):
        """Initialize the character generator."""
        self.personality_manager = None  # Will be injected if available

    def set_personality_manager(self, manager):
        """Set the personality manager for style-aware generation."""
        self.personality_manager = manager

    def generate_character(
        self,
        system: str = "D&D 5e",
        level: int = 1,
        character_class: Optional[str] = None,
        race: Optional[str] = None,
        name: Optional[str] = None,
        backstory_hints: Optional[str] = None,
        stat_generation: str = "standard",  # "standard", "random", "point_buy"
        genre: Optional[str] = None,  # Genre override
        use_extended: bool = False,  # Use ExtendedCharacter with genre data
    ) -> Character:
        """
        Generate a complete player character.

        Args:
            system: Game system (e.g., "D&D 5e")
            level: Character level
            character_class: Specific class or None for random
            race: Specific race or None for random
            name: Character name or None for generation
            backstory_hints: Hints for backstory generation
            stat_generation: Method for generating stats

        Returns:
            Complete Character object
        """
        # Validate input parameters
        param_errors = CharacterValidator.validate_generation_params(
            level=level, system=system, character_class=character_class, race=race
        )
        if param_errors:
            raise ValidationError(f"Invalid parameters: {'; '.join(param_errors)}")

        logger.info(f"Generating character for {system} at level {level}")

        # Determine genre
        if genre:
            try:
                genre_enum = TTRPGGenre(genre.lower())
            except ValueError:
                genre_enum = TTRPGGenre.FANTASY
        else:
            genre_enum = self._determine_genre_from_system(system)

        # Create base character or extended character
        if use_extended or genre_enum != TTRPGGenre.FANTASY:
            character = ExtendedCharacter(
                system=system,
                name=name or self._generate_name(race, genre_enum),
                genre=genre_enum
            )
        else:
            character = Character(
                system=system,
                name=name or self._generate_name(race, genre_enum),
                genre=genre_enum
            )

        # Set class and race based on genre
        character.character_class = self._select_class(character_class, genre_enum)
        character.race = self._select_race(race, genre_enum)

        # Generate stats
        character.stats = self._generate_stats(
            character.character_class, character.race, level, stat_generation
        )

        # Generate equipment
        character.equipment = self._generate_equipment(character.character_class, level)

        # Generate skills and proficiencies
        self._generate_skills_and_proficiencies(character)

        # Generate languages
        character.languages = self._generate_languages(character.race)

        # Generate features
        character.features = self._generate_class_features(character.character_class, level)

        # Generate spells if applicable
        if self._is_spellcaster(character.character_class):
            character.spells = self._generate_spells(character.character_class, level)

        # Set alignment
        character.alignment = self._generate_alignment()

        # Generate backstory (basic, will be enhanced by BackstoryGenerator)
        character.backstory = self._generate_basic_backstory(character, backstory_hints)

        # Generate genre-specific data if using ExtendedCharacter
        if isinstance(character, ExtendedCharacter):
            self._generate_genre_specific_data(character)

        # Validate the generated character
        validation_errors = CharacterValidator.validate_character(character)
        if validation_errors:
            logger.warning(f"Character validation issues: {validation_errors}")
            # Try to fix critical errors
            for error in validation_errors:
                if "must have stats" in error:
                    raise ValidationError(f"Critical validation error: {error}")

        logger.info(f"Character generation complete: {character.name}")
        return character

    def _determine_genre_from_system(self, system: str) -> TTRPGGenre:
        """Determine genre based on system name."""
        system_lower = system.lower()
        
        if any(term in system_lower for term in ["d&d", "pathfinder", "dungeon", "dragon"]):
            return TTRPGGenre.FANTASY
        elif any(term in system_lower for term in ["cyberpunk", "shadowrun", "chrome"]):
            return TTRPGGenre.CYBERPUNK
        elif any(term in system_lower for term in ["traveller", "stars", "space", "galaxy"]):
            return TTRPGGenre.SCI_FI
        elif any(term in system_lower for term in ["cthulhu", "delta green", "horror"]):
            return TTRPGGenre.COSMIC_HORROR
        elif any(term in system_lower for term in ["apocalypse", "fallout", "wasteland"]):
            return TTRPGGenre.POST_APOCALYPTIC
        elif any(term in system_lower for term in ["masks", "hero", "super", "marvel", "dc"]):
            return TTRPGGenre.SUPERHERO
        elif any(term in system_lower for term in ["deadlands", "western", "frontier"]):
            return TTRPGGenre.WESTERN
        else:
            return TTRPGGenre.FANTASY

    def _select_class(self, class_name: Optional[str], genre: TTRPGGenre) -> CharacterClass:
        """Select a character class based on genre."""
        if class_name:
            try:
                return CharacterClass(class_name.lower().replace(" ", "_").replace("-", "_"))
            except ValueError:
                # Custom class
                return CharacterClass.CUSTOM

        # Genre-specific class pools
        genre_classes = {
            TTRPGGenre.FANTASY: [
                CharacterClass.FIGHTER, CharacterClass.WIZARD, CharacterClass.CLERIC,
                CharacterClass.ROGUE, CharacterClass.RANGER, CharacterClass.PALADIN,
                CharacterClass.BARBARIAN, CharacterClass.SORCERER, CharacterClass.WARLOCK,
                CharacterClass.DRUID, CharacterClass.MONK, CharacterClass.BARD,
                CharacterClass.ARTIFICER
            ],
            TTRPGGenre.SCI_FI: [
                CharacterClass.ENGINEER, CharacterClass.SCIENTIST, CharacterClass.PILOT,
                CharacterClass.MARINE, CharacterClass.DIPLOMAT, CharacterClass.XENOBIOLOGIST,
                CharacterClass.TECH_SPECIALIST, CharacterClass.PSION, CharacterClass.BOUNTY_HUNTER
            ],
            TTRPGGenre.CYBERPUNK: [
                CharacterClass.NETRUNNER, CharacterClass.SOLO, CharacterClass.FIXER,
                CharacterClass.CORPORATE, CharacterClass.ROCKERBOY, CharacterClass.TECHIE,
                CharacterClass.MEDIA, CharacterClass.COP, CharacterClass.NOMAD
            ],
            TTRPGGenre.COSMIC_HORROR: [
                CharacterClass.INVESTIGATOR, CharacterClass.SCHOLAR, CharacterClass.ANTIQUARIAN,
                CharacterClass.OCCULTIST, CharacterClass.ALIENIST, CharacterClass.ARCHAEOLOGIST,
                CharacterClass.JOURNALIST, CharacterClass.DETECTIVE, CharacterClass.PROFESSOR
            ],
            TTRPGGenre.POST_APOCALYPTIC: [
                CharacterClass.SURVIVOR, CharacterClass.SCAVENGER, CharacterClass.RAIDER,
                CharacterClass.MEDIC, CharacterClass.MECHANIC, CharacterClass.TRADER,
                CharacterClass.WARLORD, CharacterClass.MUTANT_HUNTER, CharacterClass.VAULT_DWELLER
            ],
            TTRPGGenre.SUPERHERO: [
                CharacterClass.VIGILANTE, CharacterClass.POWERED, CharacterClass.GENIUS,
                CharacterClass.MARTIAL_ARTIST, CharacterClass.MYSTIC, CharacterClass.ALIEN_HERO,
                CharacterClass.TECH_HERO, CharacterClass.SIDEKICK
            ],
            TTRPGGenre.WESTERN: [
                CharacterClass.GUNSLINGER, CharacterClass.LAWMAN, CharacterClass.OUTLAW,
                CharacterClass.GAMBLER, CharacterClass.PREACHER, CharacterClass.PROSPECTOR,
                CharacterClass.NATIVE_SCOUT
            ],
        }

        # Get appropriate class pool for genre
        class_pool = genre_classes.get(genre, genre_classes[TTRPGGenre.FANTASY])
        return random.choice(class_pool)

    def _select_race(self, race_name: Optional[str], genre: TTRPGGenre) -> CharacterRace:
        """Select a character race based on genre."""
        if race_name:
            try:
                # Replace spaces and hyphens with underscores to match enum values
                normalized_race = race_name.lower().replace(" ", "_").replace("-", "_")
                return CharacterRace(normalized_race)
            except ValueError:
                # Custom race
                return CharacterRace.CUSTOM

        # Genre-specific race pools
        genre_races = {
            TTRPGGenre.FANTASY: [
                CharacterRace.HUMAN, CharacterRace.ELF, CharacterRace.DWARF,
                CharacterRace.HALFLING, CharacterRace.ORC, CharacterRace.TIEFLING,
                CharacterRace.DRAGONBORN, CharacterRace.GNOME, CharacterRace.HALF_ELF,
                CharacterRace.HALF_ORC
            ],
            TTRPGGenre.SCI_FI: [
                CharacterRace.TERRAN, CharacterRace.MARTIAN, CharacterRace.BELTER,
                CharacterRace.CYBORG, CharacterRace.ANDROID, CharacterRace.AI_CONSTRUCT,
                CharacterRace.GREY_ALIEN, CharacterRace.REPTILIAN, CharacterRace.INSECTOID,
                CharacterRace.ENERGY_BEING, CharacterRace.SILICON_BASED, CharacterRace.UPLIFTED_ANIMAL
            ],
            TTRPGGenre.CYBERPUNK: [
                CharacterRace.AUGMENTED_HUMAN, CharacterRace.FULL_CONVERSION_CYBORG,
                CharacterRace.BIOENGINEERED, CharacterRace.CLONE, CharacterRace.DIGITAL_CONSCIOUSNESS,
                CharacterRace.HUMAN
            ],
            TTRPGGenre.COSMIC_HORROR: [
                CharacterRace.HUMAN, CharacterRace.DEEP_ONE_HYBRID, CharacterRace.GHOUL,
                CharacterRace.DREAMLANDS_NATIVE, CharacterRace.TOUCHED
            ],
            TTRPGGenre.POST_APOCALYPTIC: [
                CharacterRace.PURE_STRAIN_HUMAN, CharacterRace.MUTANT,
                CharacterRace.GHOUL_WASTELANDER, CharacterRace.SYNTHETIC,
                CharacterRace.HYBRID, CharacterRace.RADIANT
            ],
            TTRPGGenre.SUPERHERO: [
                CharacterRace.HUMAN, CharacterRace.METAHUMAN, CharacterRace.INHUMAN,
                CharacterRace.ATLANTEAN, CharacterRace.AMAZONIAN, CharacterRace.KRYPTONIAN,
                CharacterRace.ASGARDIAN
            ],
            TTRPGGenre.WESTERN: [
                CharacterRace.HUMAN  # Western is typically human-only
            ],
        }

        # Get appropriate race pool for genre
        race_pool = genre_races.get(genre, genre_races[TTRPGGenre.FANTASY])
        return random.choice(race_pool)

    def _generate_stats(
        self, character_class: CharacterClass, race: CharacterRace, level: int, method: str
    ) -> CharacterStats:
        """Generate character statistics."""
        stats = CharacterStats(level=level)

        if method == "standard":
            # Use standard array
            values = self.STANDARD_ARRAY.copy()
            random.shuffle(values)

            # Assign to stats prioritizing class needs
            primary_stats = self.CLASS_PRIMARY_STATS.get(character_class, ["strength", "dexterity"])

            # Assign highest values to primary stats
            stats.strength = values.pop(0) if "strength" in primary_stats else values.pop()
            stats.dexterity = values.pop(0) if "dexterity" in primary_stats else values.pop()
            stats.constitution = values.pop(0) if "constitution" in primary_stats else values.pop()
            stats.intelligence = values.pop(0) if "intelligence" in primary_stats else values.pop()
            stats.wisdom = values.pop(0) if "wisdom" in primary_stats else values.pop()
            stats.charisma = values.pop(0) if "charisma" in primary_stats else values.pop()

        elif method == "random":
            # Roll 4d6, drop lowest
            for stat_name in [
                "strength",
                "dexterity",
                "constitution",
                "intelligence",
                "wisdom",
                "charisma",
            ]:
                rolls = sorted([random.randint(1, 6) for _ in range(4)], reverse=True)
                setattr(stats, stat_name, sum(rolls[:3]))

        else:  # point_buy
            # Start with base 8s and distribute 27 points
            stats.strength = 8
            stats.dexterity = 8
            stats.constitution = 8
            stats.intelligence = 8
            stats.wisdom = 8
            stats.charisma = 8

            points = 27
            stat_names = [
                "strength",
                "dexterity",
                "constitution",
                "intelligence",
                "wisdom",
                "charisma",
            ]

            while points > 0:
                stat = random.choice(stat_names)
                current = getattr(stats, stat)
                if current < 15:
                    # D&D 5e point buy: Increasing a score from 8 to 13 costs 1 point per increment,
                    # and from 13 to 15 costs 2 points per increment.
                    cost = 1 if current < 13 else 2
                    if points >= cost:
                        setattr(stats, stat, current + 1)
                        points -= cost

        # Apply racial bonuses
        self._apply_racial_bonuses(stats, race)

        # Calculate derived stats
        stats.proficiency_bonus = 2 + (level - 1) // 4

        # Calculate HP
        hit_die = self.HIT_DICE.get(character_class, 8)
        con_mod = stats.get_modifier(stats.constitution)
        stats.max_hit_points = hit_die + con_mod
        for _ in range(1, level):
            stats.max_hit_points += max(1, (hit_die // 2 + 1) + con_mod)
        stats.hit_points = stats.max_hit_points

        # Calculate AC (base, will be modified by equipment)
        stats.armor_class = 10 + stats.get_modifier(stats.dexterity)

        # Initiative
        stats.initiative_bonus = stats.get_modifier(stats.dexterity)

        return stats

    def _apply_racial_bonuses(self, stats: CharacterStats, race: CharacterRace):
        """Apply racial stat bonuses."""
        bonuses = self.RACIAL_BONUSES.get(race, {})

        for stat, bonus in bonuses.items():
            if stat == "all":
                # Apply to all stats
                for stat_name in [
                    "strength",
                    "dexterity",
                    "constitution",
                    "intelligence",
                    "wisdom",
                    "charisma",
                ]:
                    current = getattr(stats, stat_name)
                    setattr(stats, stat_name, current + bonus)
            elif stat == "any_two":
                # Apply to two random stats
                stat_names = [
                    "strength",
                    "dexterity",
                    "constitution",
                    "intelligence",
                    "wisdom",
                    "charisma",
                ]
                chosen = random.sample(stat_names, 2)
                for stat_name in chosen:
                    current = getattr(stats, stat_name)
                    setattr(stats, stat_name, current + bonus)
            else:
                # Apply to specific stat
                current = getattr(stats, stat)
                setattr(stats, stat, current + bonus)

    def _generate_equipment(self, character_class: CharacterClass, level: int) -> Equipment:
        """Generate starting equipment."""
        equipment = Equipment()

        # Get class-specific equipment
        class_gear = self.CLASS_EQUIPMENT.get(character_class, {})
        equipment.weapons = class_gear.get("weapons", []).copy()
        equipment.armor = class_gear.get("armor", []).copy()
        equipment.items = class_gear.get("items", []).copy()

        # Add basic adventuring gear
        equipment.items.extend(["Rope (50 ft)", "Torches (10)", "Waterskin", "Rations (5 days)"])

        # Starting gold
        equipment.currency["gold"] = random.randint(10, 50) * level

        # Add magic items for higher level characters
        if level >= 5:
            equipment.magic_items.append(self._generate_magic_item(level))

        return equipment

    def _generate_magic_item(self, level: int) -> str:
        """Generate a level-appropriate magic item."""
        if level < 5:
            return random.choice(["Potion of Healing", "+1 Weapon", "Cloak of Elvenkind"])
        elif level < 10:
            return random.choice(["+2 Weapon", "Bag of Holding", "Boots of Speed"])
        else:
            return random.choice(["+3 Weapon", "Ring of Protection", "Cloak of Displacement"])

    def _generate_skills_and_proficiencies(self, character: Character):
        """Generate skills and proficiencies based on class."""
        # Class-specific proficiencies
        class_proficiencies = {
            CharacterClass.FIGHTER: ["Athletics", "Intimidation"],
            CharacterClass.WIZARD: ["Arcana", "Investigation"],
            CharacterClass.ROGUE: ["Stealth", "Sleight of Hand", "Perception"],
            CharacterClass.CLERIC: ["Religion", "Medicine"],
            CharacterClass.RANGER: ["Survival", "Nature", "Animal Handling"],
            CharacterClass.BARBARIAN: ["Athletics", "Survival"],
            CharacterClass.PALADIN: ["Religion", "Persuasion"],
            CharacterClass.SORCERER: ["Arcana", "Persuasion"],
            CharacterClass.WARLOCK: ["Arcana", "Deception"],
            CharacterClass.DRUID: ["Nature", "Medicine"],
            CharacterClass.MONK: ["Acrobatics", "Insight"],
            CharacterClass.BARD: ["Performance", "Persuasion", "Deception"],
            CharacterClass.ARTIFICER: ["Arcana", "Investigation"],
        }

        character.proficiencies = class_proficiencies.get(
            character.character_class, ["Athletics", "Perception"]
        ).copy()

        # Calculate skill bonuses
        for skill in character.proficiencies:
            character.skills[skill] = character.stats.proficiency_bonus

    def _generate_languages(self, race: CharacterRace) -> List[str]:
        """Generate known languages based on race."""
        languages = ["Common"]

        race_languages = {
            CharacterRace.ELF: ["Elvish"],
            CharacterRace.DWARF: ["Dwarvish"],
            CharacterRace.HALFLING: ["Halfling"],
            CharacterRace.ORC: ["Orcish"],
            CharacterRace.TIEFLING: ["Infernal"],
            CharacterRace.DRAGONBORN: ["Draconic"],
            CharacterRace.GNOME: ["Gnomish"],
            CharacterRace.HALF_ELF: ["Elvish"],
            CharacterRace.HALF_ORC: ["Orcish"],
        }

        if race in race_languages:
            languages.extend(race_languages[race])

        return languages

    def _generate_class_features(self, character_class: CharacterClass, level: int) -> List[str]:
        """Generate class features based on level."""
        features = []

        # Level 1 features
        class_features = {
            CharacterClass.FIGHTER: ["Fighting Style", "Second Wind"],
            CharacterClass.WIZARD: ["Spellcasting", "Arcane Recovery"],
            CharacterClass.ROGUE: ["Sneak Attack", "Thieves' Cant"],
            CharacterClass.CLERIC: ["Spellcasting", "Divine Domain"],
            CharacterClass.RANGER: ["Favored Enemy", "Natural Explorer"],
            CharacterClass.BARBARIAN: ["Rage", "Unarmored Defense"],
            CharacterClass.PALADIN: ["Divine Sense", "Lay on Hands"],
            CharacterClass.SORCERER: ["Spellcasting", "Sorcerous Origin"],
            CharacterClass.WARLOCK: ["Otherworldly Patron", "Pact Magic"],
            CharacterClass.DRUID: ["Druidic", "Spellcasting"],
            CharacterClass.MONK: ["Unarmored Defense", "Martial Arts"],
            CharacterClass.BARD: ["Bardic Inspiration", "Spellcasting"],
            CharacterClass.ARTIFICER: ["Magical Tinkering", "Spellcasting"],
        }

        features.extend(class_features.get(character_class, []))

        # Add level-based features (simplified)
        if level >= 2:
            features.append(f"Level {level} Class Feature")
        if level >= 5:
            features.append(
                "Extra Attack"
                if character_class
                in [
                    CharacterClass.FIGHTER,
                    CharacterClass.RANGER,
                    CharacterClass.PALADIN,
                    CharacterClass.BARBARIAN,
                ]
                else "Level 5 Feature"
            )

        return features

    def _is_spellcaster(self, character_class: CharacterClass) -> bool:
        """Check if the class is a spellcaster."""
        spellcasters = [
            CharacterClass.WIZARD,
            CharacterClass.CLERIC,
            CharacterClass.SORCERER,
            CharacterClass.WARLOCK,
            CharacterClass.DRUID,
            CharacterClass.BARD,
            CharacterClass.PALADIN,
            CharacterClass.RANGER,
            CharacterClass.ARTIFICER,
        ]
        return character_class in spellcasters

    def _generate_spells(self, character_class: CharacterClass, level: int) -> List[str]:
        """Generate known/prepared spells."""
        spells = []

        # Cantrips
        cantrips = {
            CharacterClass.WIZARD: ["Fire Bolt", "Mage Hand", "Prestidigitation"],
            CharacterClass.CLERIC: ["Sacred Flame", "Guidance"],
            CharacterClass.SORCERER: ["Fire Bolt", "Minor Illusion"],
            CharacterClass.WARLOCK: ["Eldritch Blast", "Minor Illusion"],
            CharacterClass.DRUID: ["Druidcraft", "Produce Flame"],
            CharacterClass.BARD: ["Vicious Mockery", "Minor Illusion"],
        }

        if character_class in cantrips:
            spells.extend(cantrips[character_class])

        # Level 1 spells
        if level >= 1:
            level_1_spells = {
                CharacterClass.WIZARD: ["Magic Missile", "Shield", "Detect Magic"],
                CharacterClass.CLERIC: ["Cure Wounds", "Bless", "Guiding Bolt"],
                CharacterClass.SORCERER: ["Magic Missile", "Shield"],
                CharacterClass.WARLOCK: ["Hex", "Armor of Agathys"],
                CharacterClass.DRUID: ["Entangle", "Cure Wounds"],
                CharacterClass.BARD: ["Charm Person", "Healing Word"],
                CharacterClass.PALADIN: ["Divine Favor", "Cure Wounds"],
                CharacterClass.RANGER: ["Hunter's Mark", "Cure Wounds"],
            }

            if character_class in level_1_spells:
                spells.extend(level_1_spells[character_class])

        return spells

    def _generate_alignment(self) -> str:
        """Generate a random alignment."""
        ethics = ["Lawful", "Neutral", "Chaotic"]
        morals = ["Good", "Neutral", "Evil"]

        ethic = random.choice(ethics)
        moral = random.choice(morals)

        if ethic == "Neutral" and moral == "Neutral":
            return "True Neutral"

        return f"{ethic} {moral}"

    def _generate_name(self, race: Optional[CharacterRace], genre: TTRPGGenre = TTRPGGenre.FANTASY) -> str:
        """Generate a character name based on race and genre."""
        # Genre-specific name pools
        if genre == TTRPGGenre.CYBERPUNK:
            first_names = ["Neon", "Chrome", "Shadow", "Razor", "Ghost", "Binary"]
            last_names = ["Runner", "Jack", "Burn", "Wire", "Crash", "Hex"]
        elif genre == TTRPGGenre.SCI_FI:
            first_names = ["Nova", "Orion", "Luna", "Atlas", "Stellar", "Vega"]
            last_names = ["Stardust", "Cosmos", "Nebula", "Void", "Quasar", "Pulsar"]
        elif genre == TTRPGGenre.COSMIC_HORROR:
            first_names = ["Randolph", "Herbert", "Wilbur", "Lavinia", "Silas", "Ephraim"]
            last_names = ["Whateley", "Armitage", "Marsh", "Gilman", "Ward", "Carter"]
        elif genre == TTRPGGenre.POST_APOCALYPTIC:
            first_names = ["Ash", "Rust", "Storm", "Dust", "Raven", "Scar"]
            last_names = ["Walker", "Survivor", "Scav", "Hunter", "Wanderer", "Keeper"]
        elif genre == TTRPGGenre.WESTERN:
            first_names = ["Jesse", "Wyatt", "Doc", "Belle", "Calamity", "Bill"]
            last_names = ["Morgan", "Earp", "Holliday", "Starr", "James", "Cassidy"]
        elif genre == TTRPGGenre.SUPERHERO:
            first_names = ["Max", "Diana", "Victor", "Jean", "Bruce", "Clark"]
            last_names = ["Power", "Steel", "Storm", "Phoenix", "Knight", "Swift"]
        else:  # Fantasy default
            first_names = {
                CharacterRace.HUMAN: ["John", "Sarah", "Marcus", "Elena"],
                CharacterRace.ELF: ["Legolas", "Arwen", "Elrond", "Galadriel"],
                CharacterRace.DWARF: ["Thorin", "Gimli", "Dwalin", "Balin"],
                CharacterRace.HALFLING: ["Frodo", "Sam", "Pippin", "Merry"],
                None: ["Alex", "Morgan", "Jordan", "Casey"],
            }
            last_names = {
                CharacterRace.HUMAN: ["Smith", "Walker", "Hunter", "Miller"],
                CharacterRace.ELF: ["Greenleaf", "Starweaver", "Moonshadow"],
                CharacterRace.DWARF: ["Ironforge", "Stonebeard", "Battlehammer"],
                CharacterRace.HALFLING: ["Baggins", "Took", "Brandybuck"],
                None: ["Stone", "River", "Hill", "Wood"],
            }
            first = random.choice(first_names.get(race, first_names[None]))
            last = random.choice(last_names.get(race, last_names[None]))
            return f"{first} {last}"

        # For non-fantasy genres
        first = random.choice(first_names)
        last = random.choice(last_names)
        return f"{first} {last}"

    def _generate_genre_specific_data(self, character: ExtendedCharacter):
        """Generate genre-specific data for ExtendedCharacter."""
        genre = character.genre
        
        if genre == TTRPGGenre.CYBERPUNK:
            # Generate augmentations
            num_augs = random.randint(1, 3)
            for _ in range(num_augs):
                aug = CyberpunkAugmentation(
                    name=random.choice(["Neural Interface", "Cybereye", "Reflex Booster", "Subdermal Armor"]),
                    type=random.choice(["neural", "cyberlimb", "bioware", "nanotech"]),
                    quality=random.choice(["street", "military", "corporate"]),
                    humanity_cost=random.randint(1, 10),
                    description="Enhanced capability",
                    abilities=["Enhanced perception", "Increased reflexes"]
                )
                character.genre_data.augmentations.append(aug)
            character.genre_data.street_cred = random.randint(0, 10)
            
        elif genre == TTRPGGenre.SCI_FI:
            # Generate tech
            tech = SciFiTechnology(
                name=random.choice(["Plasma Rifle", "Force Shield", "Quantum Scanner"]),
                tech_level=random.randint(5, 9),
                category=random.choice(["weapon", "armor", "tool"]),
                power_source=random.choice(["fusion", "antimatter", "zero-point"]),
                description="Advanced technology",
                capabilities=["Long range", "High precision"]
            )
            character.genre_data.technologies.append(tech)
            character.genre_data.ship_assignment = random.choice(["USS Enterprise", "Nostromo", "Serenity"])
            character.genre_data.clearance_level = random.choice(["Alpha", "Beta", "Gamma", "Delta"])
            
        elif genre == TTRPGGenre.COSMIC_HORROR:
            # Generate sanity
            character.genre_data.sanity = CosmicHorrorSanity(
                current_sanity=random.randint(40, 90),
                max_sanity=100,
                indefinite_insanity=False,
                phobias=[random.choice(["Darkness", "Water", "Heights", "Crowds"])],
                manias=[],
                encounters=[],
                forbidden_knowledge=[]
            )
            
        elif genre == TTRPGGenre.POST_APOCALYPTIC:
            # Generate mutations
            if character.race == CharacterRace.MUTANT:
                mutation = PostApocalypticMutation(
                    name=random.choice(["Radiation Resistance", "Night Vision", "Extra Limb", "Acid Spit"]),
                    type=random.choice(["physical", "mental", "metabolic", "sensory"]),
                    severity=random.choice(["minor", "major"]),
                    description="Mutation from radiation exposure",
                    benefits=["Enhanced ability"],
                    drawbacks=["Social stigma"]
                )
                character.genre_data.mutations.append(mutation)
            character.genre_data.radiation_resistance = random.randint(0, 100)
            character.genre_data.survival_skills = ["Scavenging", "Water Finding", "Shelter Building"]
            
        elif genre == TTRPGGenre.SUPERHERO:
            # Generate powers
            if character.character_class == CharacterClass.POWERED:
                power = SuperheroPower(
                    name=random.choice(["Super Strength", "Flight", "Telepathy", "Energy Blast"]),
                    category=random.choice(["physical", "energy", "mental"]),
                    power_level=random.randint(5, 10),
                    origin=random.choice(["mutation", "accident", "alien"]),
                    description="Superhuman ability",
                    abilities=["Enhanced capability"],
                    limitations=["Requires concentration", "Limited uses per day"]
                )
                character.genre_data.powers.append(power)
            character.genre_data.secret_identity = f"{character.name} (civilian)"
            
        elif genre == TTRPGGenre.WESTERN:
            # Generate western data
            character.genre_data.reputation = random.choice(["Unknown", "Greenhorn", "Known", "Famous"])
            character.genre_data.bounty = random.randint(0, 5000)
            character.genre_data.quick_draw = random.randint(1, 10)

    def _generate_basic_backstory(self, character: Character, hints: Optional[str]) -> Backstory:
        """Generate a basic backstory (will be enhanced by BackstoryGenerator)."""
        backstory = Backstory()

        # Basic personality traits
        traits = [
            "I always have a plan for what to do when things go wrong.",
            "I am always calm, no matter what the situation.",
            "I enjoy being strong and like breaking things.",
            "I have a joke for every occasion.",
            "I'm driven by a wanderlust.",
        ]
        backstory.personality_traits = [random.choice(traits)]

        # Basic ideals
        ideals = [
            "Freedom. Chains are meant to be broken.",
            "Charity. I always try to help those in need.",
            "Power. I will do whatever it takes to become powerful.",
            "Honor. I don't steal from others in the trade.",
            "Logic. Emotions must not cloud our logical thinking.",
        ]
        backstory.ideals = [random.choice(ideals)]

        # Basic bonds
        bonds = [
            "I have a family, but I have no idea where they are.",
            "An injury to the unspoiled wilderness is an injury to me.",
            "I will bring terrible wrath down on the evildoers.",
            "I'm trying to pay off an old debt.",
            "I protect those who cannot protect themselves.",
        ]
        backstory.bonds = [random.choice(bonds)]

        # Basic flaws
        flaws = [
            "I judge others harshly, and myself even more severely.",
            "I'd rather eat my armor than admit I'm wrong.",
            "I have trouble trusting in my allies.",
            "I'm too greedy for my own good.",
            "I have a weakness for the vices of the city.",
        ]
        backstory.flaws = [random.choice(flaws)]

        # Basic background
        backgrounds = [
            "Acolyte",
            "Criminal",
            "Folk Hero",
            "Noble",
            "Sage",
            "Soldier",
            "Charlatan",
            "Entertainer",
            "Guild Artisan",
            "Hermit",
            "Outlander",
            "Sailor",
        ]
        backstory.background = random.choice(backgrounds)

        # Origin based on race
        if character.race == CharacterRace.ELF:
            backstory.origin = "the ancient elven forests"
        elif character.race == CharacterRace.DWARF:
            backstory.origin = "the mountain strongholds"
        elif character.race == CharacterRace.HALFLING:
            backstory.origin = "a peaceful shire village"
        else:
            backstory.origin = "a small frontier town"

        # Basic motivation
        backstory.motivation = f"Seeking adventure as a {character.get_class_name()}"

        # Basic goals
        backstory.goals = ["Find fortune and glory", "Master my abilities"]

        return backstory
