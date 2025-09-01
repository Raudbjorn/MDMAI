"""NPC generation system for TTRPG Assistant."""

import logging
import random
from typing import List, Optional

from .backstory_generator import BackstoryGenerator
from .character_generator import CharacterGenerator
from .models import (
    NPC,
    CharacterBackground,
    CharacterClass,
    CharacterMotivation,
    CharacterRace,
    CharacterStats,
    CharacterTrait,
    Equipment,
    ItemType,
    NPCRole,
    PersonalityTrait,
    StoryHook,
    WeaponType,
    WorldElement,
)
from .validators import CharacterValidator, ValidationError

logger = logging.getLogger(__name__)


class NPCGenerator:
    """Generate NPCs with enriched traits, motivations, and diverse characteristics."""
    
    @classmethod
    def get_random_npc_traits(cls, count: int = 3) -> List[CharacterTrait]:
        """Get random NPC traits from enriched content."""
        # NPCs often have more pronounced traits
        trait_categories = {
            'physical': [t for t in CharacterTrait if t.name in ['SCARRED', 'WEATHERED', 'MUSCULAR', 'SLENDER', 'STOCKY']],
            'mental': [t for t in CharacterTrait if t.name in ['CUNNING', 'SHREWD', 'OBSERVANT', 'KNOWLEDGEABLE']],
            'emotional': [t for t in CharacterTrait if t.name in ['GRIM', 'CHEERFUL', 'SUSPICIOUS', 'FRIENDLY', 'GRUFF']],
            'social': [t for t in CharacterTrait if t.name in ['MYSTERIOUS', 'INTIMIDATING', 'CHARMING', 'RESERVED']]
        }
        
        selected = []
        for category, traits in trait_categories.items():
            if traits and len(selected) < count:
                selected.append(random.choice(traits))
        
        # Fill remaining with random traits
        while len(selected) < count:
            trait = random.choice(list(CharacterTrait))
            if trait not in selected:
                selected.append(trait)
        
        return selected[:count]
    
    @classmethod
    def get_npc_motivation(cls, role: NPCRole) -> CharacterMotivation:
        """Get appropriate motivation for NPC role."""
        role_motivations = {
            NPCRole.MERCHANT: [CharacterMotivation.WEALTH, CharacterMotivation.PROSPERITY, CharacterMotivation.STATUS],
            NPCRole.GUARD: [CharacterMotivation.DUTY, CharacterMotivation.HONOR, CharacterMotivation.PROTECTION],
            NPCRole.NOBLE: [CharacterMotivation.POWER, CharacterMotivation.LEGACY, CharacterMotivation.INFLUENCE],
            NPCRole.SCHOLAR: [CharacterMotivation.KNOWLEDGE, CharacterMotivation.DISCOVERY, CharacterMotivation.TRUTH],
            NPCRole.CRIMINAL: [CharacterMotivation.SURVIVAL, CharacterMotivation.WEALTH, CharacterMotivation.FREEDOM],
            NPCRole.PRIEST: [CharacterMotivation.SERVICE, CharacterMotivation.SALVATION, CharacterMotivation.DUTY],
            NPCRole.ADVENTURER: [CharacterMotivation.ADVENTURE, CharacterMotivation.GLORY, CharacterMotivation.WEALTH],
        }
        
        motivations = role_motivations.get(role, list(CharacterMotivation))
        return random.choice(motivations) if motivations else CharacterMotivation.SURVIVAL

    # NPC stat modifiers by role
    ROLE_STAT_MODIFIERS = {
        NPCRole.MERCHANT: {"charisma": 2, "intelligence": 1},
        NPCRole.GUARD: {"strength": 2, "constitution": 1},
        NPCRole.NOBLE: {"charisma": 2, "wisdom": 1},
        NPCRole.SCHOLAR: {"intelligence": 3, "wisdom": 1},
        NPCRole.CRIMINAL: {"dexterity": 2, "charisma": 1},
        NPCRole.INNKEEPER: {"charisma": 1, "constitution": 1},
        NPCRole.PRIEST: {"wisdom": 2, "charisma": 1},
        NPCRole.ADVENTURER: {"all": 1},
        NPCRole.ARTISAN: {"dexterity": 1, "intelligence": 1},
        NPCRole.COMMONER: {},
        NPCRole.SOLDIER: {"strength": 1, "constitution": 1},
        NPCRole.MAGE: {"intelligence": 3},
        NPCRole.ASSASSIN: {"dexterity": 3, "intelligence": 1},
        NPCRole.HEALER: {"wisdom": 2, "intelligence": 1},
    }

    # Role to class mapping for combat NPCs
    ROLE_CLASS_MAPPING = {
        NPCRole.GUARD: CharacterClass.FIGHTER,
        NPCRole.SOLDIER: CharacterClass.FIGHTER,
        NPCRole.ADVENTURER: CharacterClass.FIGHTER,  # Can vary
        NPCRole.MAGE: CharacterClass.WIZARD,
        NPCRole.PRIEST: CharacterClass.CLERIC,
        NPCRole.HEALER: CharacterClass.CLERIC,
        NPCRole.CRIMINAL: CharacterClass.ROGUE,
        NPCRole.ASSASSIN: CharacterClass.ROGUE,
        NPCRole.SCHOLAR: CharacterClass.WIZARD,
    }

    # Personality trait categories by role
    ROLE_PERSONALITY_TRAITS = {
        NPCRole.MERCHANT: {
            "demeanor": ["shrewd", "friendly", "calculating", "enthusiastic"],
            "motivation": ["profit", "reputation", "connections", "survival"],
            "quirk": [
                "haggles everything",
                "knows everyone",
                "suspicious of nobles",
                "loves gossip",
            ],
        },
        NPCRole.GUARD: {
            "demeanor": ["stern", "bored", "alert", "friendly"],
            "motivation": ["duty", "pay", "promotion", "justice"],
            "quirk": ["by the book", "easily bribed", "war veteran", "new recruit"],
        },
        NPCRole.NOBLE: {
            "demeanor": ["arrogant", "gracious", "scheming", "naive"],
            "motivation": ["power", "legacy", "pleasure", "duty"],
            "quirk": ["collects oddities", "secret vice", "patron of arts", "paranoid"],
        },
        NPCRole.SCHOLAR: {
            "demeanor": ["distracted", "passionate", "condescending", "helpful"],
            "motivation": ["knowledge", "discovery", "recognition", "truth"],
            "quirk": [
                "terrible memory for names",
                "obsessed with topic",
                "speaks in quotes",
                "messy",
            ],
        },
        NPCRole.CRIMINAL: {
            "demeanor": ["paranoid", "charming", "aggressive", "professional"],
            "motivation": ["wealth", "survival", "revenge", "thrill"],
            "quirk": ["superstitious", "code of honor", "gambling problem", "loyal to crew"],
        },
        NPCRole.INNKEEPER: {
            "demeanor": ["welcoming", "gruff", "nosy", "protective"],
            "motivation": ["profit", "community", "information", "peace"],
            "quirk": ["knows all rumors", "former adventurer", "secret recipe", "matchmaker"],
        },
        NPCRole.PRIEST: {
            "demeanor": ["serene", "zealous", "compassionate", "stern"],
            "motivation": ["faith", "charity", "conversion", "redemption"],
            "quirk": ["doubting faith", "sees omens", "strict vows", "worldly past"],
        },
        NPCRole.ADVENTURER: {
            "demeanor": ["confident", "weary", "eager", "cynical"],
            "motivation": ["glory", "wealth", "justice", "wanderlust"],
            "quirk": ["tells tall tales", "looking for party", "retired", "cursed"],
        },
        NPCRole.ARTISAN: {
            "demeanor": ["proud", "perfectionist", "creative", "business-minded"],
            "motivation": ["mastery", "legacy", "innovation", "wealth"],
            "quirk": [
                "never satisfied",
                "rival artisan",
                "secret technique",
                "artistic temperament",
            ],
        },
        NPCRole.COMMONER: {
            "demeanor": ["humble", "fearful", "friendly", "bitter"],
            "motivation": ["survival", "family", "comfort", "escape"],
            "quirk": ["local expert", "superstitious", "gossip", "ambitious"],
        },
        NPCRole.SOLDIER: {
            "demeanor": ["disciplined", "battle-hardened", "loyal", "disillusioned"],
            "motivation": ["duty", "honor", "survival", "retirement"],
            "quirk": ["war stories", "follows orders", "PTSD", "seeks glory"],
        },
        NPCRole.MAGE: {
            "demeanor": ["mysterious", "arrogant", "curious", "reclusive"],
            "motivation": ["power", "knowledge", "immortality", "balance"],
            "quirk": ["magical accident", "rival mage", "forbidden research", "magical pet"],
        },
        NPCRole.ASSASSIN: {
            "demeanor": ["cold", "professional", "charming", "invisible"],
            "motivation": ["contract", "revenge", "ideology", "thrill"],
            "quirk": ["never fails", "moral code", "double life", "marked by guild"],
        },
        NPCRole.HEALER: {
            "demeanor": ["caring", "exhausted", "practical", "mystic"],
            "motivation": ["helping others", "knowledge", "atonement", "profit"],
            "quirk": ["pacifist", "seen too much", "herbal remedies", "divine visions"],
        },
    }

    # Knowledge areas by role
    ROLE_KNOWLEDGE = {
        NPCRole.MERCHANT: ["trade routes", "market prices", "local politics", "customer gossip"],
        NPCRole.GUARD: ["local law", "criminal activity", "city layout", "recent incidents"],
        NPCRole.NOBLE: [
            "court intrigue",
            "family histories",
            "political alliances",
            "social customs",
        ],
        NPCRole.SCHOLAR: ["ancient history", "arcane theory", "local legends", "rare texts"],
        NPCRole.CRIMINAL: ["black market", "guard patrols", "hideouts", "underworld politics"],
        NPCRole.INNKEEPER: ["local rumors", "traveler news", "town history", "regular customers"],
        NPCRole.PRIEST: ["religious lore", "divine signs", "moral guidance", "community needs"],
        NPCRole.ADVENTURER: [
            "dungeon locations",
            "monster lore",
            "treasure rumors",
            "survival tips",
        ],
        NPCRole.ARTISAN: [
            "craft techniques",
            "material sources",
            "guild politics",
            "customer preferences",
        ],
        NPCRole.COMMONER: ["local gossip", "daily life", "town personalities", "practical skills"],
        NPCRole.SOLDIER: [
            "military tactics",
            "enemy movements",
            "fortifications",
            "command structure",
        ],
        NPCRole.MAGE: [
            "magical theory",
            "spell components",
            "planar knowledge",
            "ancient artifacts",
        ],
        NPCRole.ASSASSIN: ["target patterns", "poison craft", "escape routes", "guild protocols"],
        NPCRole.HEALER: ["ailments", "herb lore", "anatomy", "local health issues"],
    }

    # Combat behavior by role
    ROLE_COMBAT_BEHAVIOR = {
        NPCRole.GUARD: "defensive",
        NPCRole.SOLDIER: "tactical",
        NPCRole.ADVENTURER: "tactical",
        NPCRole.CRIMINAL: "opportunistic",
        NPCRole.ASSASSIN: "aggressive",
        NPCRole.MAGE: "ranged",
        NPCRole.PRIEST: "supportive",
        NPCRole.HEALER: "avoidant",
        NPCRole.NOBLE: "flee",
        NPCRole.COMMONER: "flee",
        NPCRole.MERCHANT: "negotiate",
        NPCRole.SCHOLAR: "flee",
        NPCRole.INNKEEPER: "defensive",
        NPCRole.ARTISAN: "flee",
    }

    def __init__(self):
        """Initialize the NPC generator."""
        self.character_generator = CharacterGenerator()
        self.backstory_generator = BackstoryGenerator()
        self.personality_manager = None

    def set_personality_manager(self, manager):
        """Set the personality manager for style-aware generation."""
        self.personality_manager = manager
        self.character_generator.set_personality_manager(manager)
        self.backstory_generator.personality_manager = manager

    def generate_npc(
        self,
        system: str = "D&D 5e",
        role: Optional[str] = None,
        level: Optional[int] = None,
        name: Optional[str] = None,
        personality_traits: Optional[List[str]] = None,
        importance: str = "minor",  # "minor", "supporting", "major"
        party_level: Optional[int] = None,
        backstory_depth: str = "simple",  # "simple", "standard", "detailed"
    ) -> NPC:
        """
        Generate a complete NPC.

        Args:
            system: Game system
            role: NPC role/occupation
            level: Specific level, or auto-calculate from party
            name: NPC name or None for generation
            personality_traits: Specific traits or None for generation
            importance: NPC importance level
            party_level: Party level for scaling
            backstory_depth: Level of backstory detail

        Returns:
            Complete NPC object
        """
        # Validate input parameters
        if level is not None:
            param_errors = CharacterValidator.validate_generation_params(level=level, system=system)
            if param_errors:
                raise ValidationError(f"Invalid parameters: {'; '.join(param_errors)}")

        logger.info(f"Generating {importance} NPC with role {role} for {system}")

        # Create base NPC
        npc = NPC(
            system=system,
            name=name or self._generate_npc_name(role),
            importance=importance.capitalize(),
        )

        # Set role
        npc.role = self._select_role(role)
        if npc.role == NPCRole.CUSTOM:
            npc.custom_role = role

        # Calculate appropriate level
        if level is None:
            level = self._calculate_npc_level(npc.role, importance, party_level)

        # Generate base character stats
        self._generate_base_stats(npc, level)

        # Apply role-specific modifications
        self._apply_role_modifiers(npc)

        # Generate personality with enriched traits
        npc.personality_traits = self._generate_enriched_personality_traits(npc.role, personality_traits)
        
        # Apply enriched character traits
        self._apply_enriched_traits_to_npc(npc)
        
        # Set enriched motivations
        self._set_enriched_motivations(npc)

        # Set behavioral attributes
        self._set_behavioral_attributes(npc)

        # Generate knowledge areas
        npc.knowledge_areas = self._get_knowledge_areas(npc.role)

        # Generate secrets based on importance
        if importance in ["supporting", "major"]:
            npc.secrets = self._generate_secrets(npc.role, importance)

        # Generate equipment
        npc.equipment = self._generate_npc_equipment(npc.role, level)

        # Generate skills and proficiencies
        self._generate_npc_skills(npc)

        # Generate languages
        npc.languages = self._generate_npc_languages(npc)

        # Generate backstory
        if backstory_depth != "none":
            npc.backstory = self.backstory_generator.generate_backstory(
                npc, hints=f"{npc.get_role_name()} NPC", depth=backstory_depth
            )

        # Set location and occupation
        npc.occupation = npc.get_role_name()
        npc.location = self._generate_location(npc.role)

        # Set faction if relevant
        if importance == "major" or npc.role in [NPCRole.NOBLE, NPCRole.SOLDIER, NPCRole.CRIMINAL]:
            npc.faction = self._generate_faction(npc.role)

        # Validate the generated NPC
        validation_errors = CharacterValidator.validate_npc(npc)
        if validation_errors:
            logger.warning(f"NPC validation issues: {validation_errors}")
            # Fix critical errors
            for error in validation_errors:
                if "must have" in error.lower():
                    raise ValidationError(f"Critical validation error: {error}")

        logger.info(f"NPC generation complete: {npc.name} ({npc.get_role_name()})")
        return npc

    def _select_role(self, role_name: Optional[str]) -> NPCRole:
        """Select an NPC role."""
        if role_name:
            try:
                return NPCRole(role_name.lower())
            except ValueError:
                # Custom role
                return NPCRole.CUSTOM

        # Random selection weighted by commonality
        weights = {
            NPCRole.COMMONER: 30,
            NPCRole.MERCHANT: 15,
            NPCRole.GUARD: 15,
            NPCRole.ARTISAN: 10,
            NPCRole.INNKEEPER: 5,
            NPCRole.CRIMINAL: 5,
            NPCRole.PRIEST: 5,
            NPCRole.SCHOLAR: 5,
            NPCRole.SOLDIER: 3,
            NPCRole.NOBLE: 3,
            NPCRole.ADVENTURER: 2,
            NPCRole.MAGE: 1,
            NPCRole.ASSASSIN: 0.5,
            NPCRole.HEALER: 0.5,
        }

        roles = list(weights.keys())
        probabilities = list(weights.values())
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]

        return random.choices(roles, weights=probabilities)[0]

    def _calculate_npc_level(
        self, role: NPCRole, importance: str, party_level: Optional[int]
    ) -> int:
        """Calculate appropriate NPC level."""
        if party_level is None:
            party_level = 1

        # Base level adjustments
        importance_modifiers = {"minor": -2, "supporting": 0, "major": 2}

        # Role-based level ranges
        role_levels = {
            NPCRole.COMMONER: (0, 1),
            NPCRole.MERCHANT: (0, 3),
            NPCRole.GUARD: (1, 5),
            NPCRole.ARTISAN: (1, 3),
            NPCRole.INNKEEPER: (0, 2),
            NPCRole.CRIMINAL: (1, 5),
            NPCRole.PRIEST: (1, 7),
            NPCRole.SCHOLAR: (1, 5),
            NPCRole.SOLDIER: (1, 7),
            NPCRole.NOBLE: (0, 5),
            NPCRole.ADVENTURER: (1, 10),
            NPCRole.MAGE: (3, 10),
            NPCRole.ASSASSIN: (3, 10),
            NPCRole.HEALER: (1, 7),
        }

        min_level, max_level = role_levels.get(role, (0, 5))

        # Calculate level
        base_level = party_level + importance_modifiers[importance]
        npc_level = max(min_level, min(base_level, max_level))

        # Add some randomness
        npc_level += random.randint(-1, 1)
        npc_level = max(0, npc_level)  # Can't be negative

        return npc_level

    def _generate_base_stats(self, npc: NPC, level: int):
        """Generate base statistics for the NPC."""
        # Commoners and non-combatants get simpler stats
        if npc.role in [NPCRole.COMMONER, NPCRole.MERCHANT, NPCRole.SCHOLAR, NPCRole.ARTISAN]:
            # Simple stat generation
            npc.stats = CharacterStats(
                strength=random.randint(8, 12),
                dexterity=random.randint(8, 12),
                constitution=random.randint(8, 12),
                intelligence=random.randint(8, 12),
                wisdom=random.randint(8, 12),
                charisma=random.randint(8, 12),
                level=level,
            )
        else:
            # Use character generator for combat-capable NPCs
            if npc.role in self.ROLE_CLASS_MAPPING:
                npc.character_class = self.ROLE_CLASS_MAPPING[npc.role]

            # Generate using standard array but adjusted for NPC
            base_char = self.character_generator.generate_character(
                system=npc.system,
                level=level,
                character_class=npc.character_class.value if npc.character_class else None,
                stat_generation="standard",
            )

            npc.stats = base_char.stats
            npc.skills = base_char.skills
            npc.features = base_char.features
            npc.spells = base_char.spells

        # Calculate HP based on role
        npc.stats.max_hit_points = self._calculate_npc_hp(npc.role, level, npc.stats.constitution)
        npc.stats.hit_points = npc.stats.max_hit_points

        # Set proficiency bonus
        npc.stats.proficiency_bonus = 2 + (level // 4)

    def _calculate_npc_hp(self, role: NPCRole, level: int, constitution: int) -> int:
        """Calculate NPC hit points based on role."""
        con_mod = (constitution - 10) // 2

        # Hit dice by role toughness
        hit_dice = {
            NPCRole.COMMONER: 4,
            NPCRole.MERCHANT: 4,
            NPCRole.SCHOLAR: 4,
            NPCRole.ARTISAN: 6,
            NPCRole.INNKEEPER: 6,
            NPCRole.PRIEST: 6,
            NPCRole.HEALER: 6,
            NPCRole.CRIMINAL: 8,
            NPCRole.NOBLE: 6,
            NPCRole.GUARD: 10,
            NPCRole.SOLDIER: 10,
            NPCRole.ADVENTURER: 10,
            NPCRole.MAGE: 6,
            NPCRole.ASSASSIN: 8,
        }

        die = hit_dice.get(role, 6)

        if level == 0:
            # Commoner level
            return max(1, die // 2 + con_mod)

        # Calculate HP
        hp = die + con_mod
        for _ in range(1, level):
            hp += max(1, (die // 2 + 1) + con_mod)

        return hp

    def _apply_role_modifiers(self, npc: NPC):
        """Apply role-specific stat modifiers."""
        modifiers = self.ROLE_STAT_MODIFIERS.get(npc.role, {})

        for stat, bonus in modifiers.items():
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
                    current = getattr(npc.stats, stat_name)
                    setattr(npc.stats, stat_name, current + bonus)
            else:
                # Apply to specific stat
                current = getattr(npc.stats, stat)
                setattr(npc.stats, stat, current + bonus)

        # Recalculate derived stats
        npc.stats.armor_class = 10 + npc.stats.get_modifier(npc.stats.dexterity)
        npc.stats.initiative_bonus = npc.stats.get_modifier(npc.stats.dexterity)

    def _generate_personality_traits(
        self, role: NPCRole, custom_traits: Optional[List[str]]
    ) -> List[PersonalityTrait]:
        """Generate personality traits for the NPC."""
        traits = []

        if custom_traits:
            # Use provided traits
            for trait_str in custom_traits:
                traits.append(PersonalityTrait(category="custom", trait=trait_str, description=""))
        else:
            # Generate based on role
            trait_pool = self.ROLE_PERSONALITY_TRAITS.get(
                role, self.ROLE_PERSONALITY_TRAITS[NPCRole.COMMONER]
            )

            for category, options in trait_pool.items():
                trait = random.choice(options)
                traits.append(
                    PersonalityTrait(
                        category=category,
                        trait=trait,
                        description=self._get_trait_description(category, trait),
                    )
                )

        return traits

    def _get_trait_description(self, category: str, trait: str) -> str:
        """Get a description for a personality trait."""
        descriptions = {
            "demeanor": f"Generally presents as {trait} in social interactions",
            "motivation": f"Primarily driven by {trait}",
            "quirk": f"Has a notable quirk: {trait}",
        }

        return descriptions.get(category, f"Characterized by: {trait}")

    def _set_behavioral_attributes(self, npc: NPC):
        """Set NPC behavioral attributes."""
        # Combat behavior
        npc.combat_behavior = self.ROLE_COMBAT_BEHAVIOR.get(npc.role, "defensive").capitalize()

        # Interaction style based on personality
        if any(t.trait in ["friendly", "welcoming", "helpful"] for t in npc.personality_traits):
            npc.interaction_style = "Friendly"
        elif any(t.trait in ["paranoid", "suspicious", "cold"] for t in npc.personality_traits):
            npc.interaction_style = "Suspicious"
        elif any(t.trait in ["arrogant", "condescending", "stern"] for t in npc.personality_traits):
            npc.interaction_style = "Dismissive"
        elif any(
            t.trait in ["professional", "disciplined", "business-minded"]
            for t in npc.personality_traits
        ):
            npc.interaction_style = "Professional"
        else:
            npc.interaction_style = "Neutral"

        # Attitude towards party (can be modified later)
        npc.attitude_towards_party = "Neutral"

    def _get_knowledge_areas(self, role: NPCRole) -> List[str]:
        """Get knowledge areas for the NPC role."""
        return self.ROLE_KNOWLEDGE.get(role, ["local area", "general gossip"]).copy()

    def _generate_secrets(self, role: NPCRole, importance: str) -> List[str]:
        """Generate secrets for the NPC."""
        num_secrets = 1 if importance == "supporting" else random.randint(2, 3)

        secret_templates = {
            NPCRole.MERCHANT: [
                "Smuggles contraband on the side",
                "Has connections to the thieves' guild",
                "Knows about a valuable shipment coming",
                "Owes money to dangerous people",
            ],
            NPCRole.GUARD: [
                "Takes bribes from criminals",
                "Witnessed a crime but was paid to forget",
                "Has a criminal past",
                "Secretly sympathizes with rebels",
            ],
            NPCRole.NOBLE: [
                "Has an illegitimate child",
                "Plotting against another noble house",
                "Secretly bankrupt",
                "Involved in forbidden magic",
            ],
            NPCRole.CRIMINAL: [
                "Working as a double agent",
                "Planning a major heist",
                "Has a price on their head",
                "Knows the location of hidden treasure",
            ],
            NPCRole.PRIEST: [
                "Lost their faith",
                "Stole from the church coffers",
                "Has a dark past before joining the clergy",
                "Received a disturbing divine vision",
            ],
            NPCRole.SCHOLAR: [
                "Discovered dangerous knowledge",
                "Forged important documents",
                "Knows the location of a powerful artifact",
                "Conducts forbidden research",
            ],
            NPCRole.INNKEEPER: [
                "The inn is a front for illegal activity",
                "Spies on important guests",
                "Hiding someone in the cellar",
                "Poisoned a guest once",
            ],
        }

        default_secrets = [
            "Has a hidden agenda",
            "Knows more than they let on",
            "Connected to a secret organization",
            "Hiding their true identity",
        ]

        secret_pool = secret_templates.get(role, default_secrets)
        secrets = random.sample(secret_pool, min(num_secrets, len(secret_pool)))

        return secrets

    def _generate_npc_equipment(self, role: NPCRole, level: int) -> Equipment:
        """Generate equipment appropriate for the NPC."""
        equipment = Equipment()

        # Role-specific equipment
        role_equipment = {
            NPCRole.GUARD: {
                "weapons": ["Spear", "Shortsword"],
                "armor": ["Chain Shirt", "Shield"],
                "items": ["Guard whistle", "Manacles"],
            },
            NPCRole.MERCHANT: {
                "weapons": ["Dagger"],
                "armor": [],
                "items": ["Ledger", "Merchant scales", "Sample goods"],
            },
            NPCRole.CRIMINAL: {
                "weapons": ["Dagger", "Shortsword"],
                "armor": ["Leather Armor"],
                "items": ["Thieves' tools", "Dark cloak"],
            },
            NPCRole.NOBLE: {
                "weapons": ["Ornate dagger"],
                "armor": ["Fine clothes"],
                "items": ["Signet ring", "Perfume", "Letter of introduction"],
            },
            NPCRole.PRIEST: {
                "weapons": ["Mace"],
                "armor": ["Robes"],
                "items": ["Holy symbol", "Prayer book", "Healing herbs"],
            },
            NPCRole.SCHOLAR: {
                "weapons": [],
                "armor": ["Robes"],
                "items": ["Books", "Ink and quill", "Magnifying glass"],
            },
            NPCRole.INNKEEPER: {
                "weapons": ["Club"],
                "armor": ["Apron"],
                "items": ["Keys", "Coin purse", "Guest ledger"],
            },
            NPCRole.SOLDIER: {
                "weapons": ["Longsword", "Crossbow"],
                "armor": ["Chain Mail", "Shield"],
                "items": ["Military insignia", "Rations"],
            },
            NPCRole.ADVENTURER: {
                "weapons": ["Longsword", "Shortbow"],
                "armor": ["Studded Leather"],
                "items": ["Rope", "Torch", "Adventuring gear"],
            },
            NPCRole.MAGE: {
                "weapons": ["Staff"],
                "armor": ["Robes"],
                "items": ["Spellbook", "Component pouch", "Arcane focus"],
            },
            NPCRole.ASSASSIN: {
                "weapons": ["Poisoned dagger", "Hand crossbow"],
                "armor": ["Dark leather"],
                "items": ["Poison vials", "Disguise kit", "Smoke bombs"],
            },
            NPCRole.HEALER: {
                "weapons": ["Staff"],
                "armor": ["Robes"],
                "items": ["Healer's kit", "Herbs", "Bandages"],
            },
            NPCRole.ARTISAN: {
                "weapons": ["Hammer"],
                "armor": ["Leather apron"],
                "items": ["Artisan's tools", "Raw materials", "Finished goods"],
            },
            NPCRole.COMMONER: {
                "weapons": ["Club", "Knife"],
                "armor": ["Common clothes"],
                "items": ["Random trinket", "Family heirloom"],
            },
        }

        gear = role_equipment.get(role, role_equipment[NPCRole.COMMONER])
        equipment.weapons = gear["weapons"].copy()
        equipment.armor = gear["armor"].copy()
        equipment.items = gear["items"].copy()

        # Currency based on role and level
        wealth_multipliers = {
            NPCRole.NOBLE: 100,
            NPCRole.MERCHANT: 50,
            NPCRole.CRIMINAL: 20,
            NPCRole.ADVENTURER: 30,
            NPCRole.INNKEEPER: 15,
            NPCRole.ARTISAN: 10,
            NPCRole.GUARD: 5,
            NPCRole.SOLDIER: 5,
            NPCRole.PRIEST: 3,
            NPCRole.SCHOLAR: 5,
            NPCRole.COMMONER: 1,
            NPCRole.MAGE: 20,
            NPCRole.ASSASSIN: 30,
            NPCRole.HEALER: 10,
        }

        multiplier = wealth_multipliers.get(role, 1)
        equipment.currency["gold"] = random.randint(1, 10) * multiplier * max(1, level)
        equipment.currency["silver"] = random.randint(0, 20)
        equipment.currency["copper"] = random.randint(0, 100)

        return equipment

    def _generate_npc_skills(self, npc: NPC):
        """Generate skills and proficiencies for the NPC."""
        # Role-specific skills
        role_skills = {
            NPCRole.MERCHANT: ["Persuasion", "Insight", "Deception"],
            NPCRole.GUARD: ["Perception", "Athletics", "Intimidation"],
            NPCRole.NOBLE: ["Persuasion", "History", "Intimidation"],
            NPCRole.SCHOLAR: ["History", "Arcana", "Investigation"],
            NPCRole.CRIMINAL: ["Stealth", "Deception", "Sleight of Hand"],
            NPCRole.INNKEEPER: ["Insight", "Persuasion", "Perception"],
            NPCRole.PRIEST: ["Religion", "Medicine", "Insight"],
            NPCRole.ADVENTURER: ["Survival", "Athletics", "Perception"],
            NPCRole.ARTISAN: ["Crafting", "Investigation", "Persuasion"],
            NPCRole.COMMONER: ["Animal Handling", "Survival"],
            NPCRole.SOLDIER: ["Athletics", "Intimidation", "Survival"],
            NPCRole.MAGE: ["Arcana", "History", "Investigation"],
            NPCRole.ASSASSIN: ["Stealth", "Deception", "Acrobatics"],
            NPCRole.HEALER: ["Medicine", "Nature", "Insight"],
        }

        npc.proficiencies = role_skills.get(npc.role, []).copy()

        # Calculate skill bonuses
        skill_abilities = {
            "Athletics": "strength",
            "Acrobatics": "dexterity",
            "Sleight of Hand": "dexterity",
            "Stealth": "dexterity",
            "Arcana": "intelligence",
            "History": "intelligence",
            "Investigation": "intelligence",
            "Nature": "intelligence",
            "Religion": "intelligence",
            "Animal Handling": "wisdom",
            "Insight": "wisdom",
            "Medicine": "wisdom",
            "Perception": "wisdom",
            "Survival": "wisdom",
            "Deception": "charisma",
            "Intimidation": "charisma",
            "Performance": "charisma",
            "Persuasion": "charisma",
        }

        for skill in npc.proficiencies:
            if skill in skill_abilities:
                ability = skill_abilities[skill]
                ability_score = getattr(npc.stats, ability)
                modifier = npc.stats.get_modifier(ability_score)
                npc.skills[skill] = modifier + npc.stats.proficiency_bonus

    def _generate_npc_languages(self, npc: NPC) -> List[str]:
        """Generate languages known by the NPC."""
        languages = ["Common"]

        # Additional languages based on role and intelligence
        int_mod = npc.stats.get_modifier(npc.stats.intelligence)

        if npc.role in [NPCRole.SCHOLAR, NPCRole.MAGE, NPCRole.NOBLE]:
            # Educated NPCs know more languages
            additional_languages = ["Elvish", "Dwarvish", "Draconic", "Celestial", "Infernal"]
            num_additional = min(2 + max(0, int_mod), len(additional_languages))
            languages.extend(random.sample(additional_languages, num_additional))
        elif npc.role in [NPCRole.MERCHANT, NPCRole.ADVENTURER]:
            # Travelers know trade languages
            trade_languages = ["Elvish", "Dwarvish", "Halfling"]
            languages.append(random.choice(trade_languages))
        elif npc.role == NPCRole.CRIMINAL:
            # Criminals know Thieves' Cant
            languages.append("Thieves' Cant")

        # Racial language if NPC has a race
        if npc.race:
            race_languages = {
                CharacterRace.ELF: "Elvish",
                CharacterRace.DWARF: "Dwarvish",
                CharacterRace.HALFLING: "Halfling",
                CharacterRace.ORC: "Orcish",
                CharacterRace.TIEFLING: "Infernal",
            }

            if npc.race in race_languages and race_languages[npc.race] not in languages:
                languages.append(race_languages[npc.race])

        return languages

    def _generate_npc_name(self, role: Optional[str]) -> str:
        """Generate a name appropriate for the NPC role."""
        # Role-based name styles
        first_names = {
            "common": ["Tom", "Mary", "Jack", "Sarah", "Will", "Anne"],
            "noble": ["Lord Marcus", "Lady Elena", "Sir Reginald", "Dame Victoria"],
            "criminal": ["Snake", "Whisper", "Red", "Shadow", "Blade"],
            "scholarly": ["Aldric", "Minerva", "Thaddeus", "Cordelia"],
        }

        last_names = {
            "common": ["Miller", "Smith", "Cooper", "Fletcher", "Baker"],
            "noble": ["Blackstone", "Goldshire", "Ravencrest", "Winterhold"],
            "criminal": ["the Quick", "One-Eye", "the Silent", "Fingers"],
            "scholarly": ["the Wise", "of the Tower", "Scrollkeeper", "the Learned"],
        }

        # Determine name style based on role
        if role and "noble" in role.lower():
            style = "noble"
        elif role and any(x in role.lower() for x in ["criminal", "thief", "assassin"]):
            style = "criminal"
        elif role and any(x in role.lower() for x in ["scholar", "mage", "wizard"]):
            style = "scholarly"
        else:
            style = "common"

        first = random.choice(first_names[style])
        last = random.choice(last_names[style])

        return f"{first} {last}"

    def _generate_location(self, role: NPCRole) -> str:
        """Generate an appropriate location for the NPC."""
        locations = {
            NPCRole.MERCHANT: ["Market Square", "Trade District", "Merchant Quarter"],
            NPCRole.GUARD: ["City Gates", "Watch Post", "Patrol Route"],
            NPCRole.NOBLE: ["Noble Estate", "Royal Court", "Manor House"],
            NPCRole.SCHOLAR: ["Library", "Academy", "Scholar's Tower"],
            NPCRole.CRIMINAL: ["Back Alleys", "Thieves' Den", "Underground"],
            NPCRole.INNKEEPER: ["The Inn", "Tavern", "Common Room"],
            NPCRole.PRIEST: ["Temple", "Shrine", "Chapel"],
            NPCRole.ADVENTURER: ["Tavern", "Guild Hall", "On the Road"],
            NPCRole.ARTISAN: ["Workshop", "Guild Hall", "Market Stall"],
            NPCRole.COMMONER: ["Home", "Streets", "Market"],
            NPCRole.SOLDIER: ["Barracks", "Guard Post", "Training Grounds"],
            NPCRole.MAGE: ["Tower", "Sanctum", "Arcane Academy"],
            NPCRole.ASSASSIN: ["Unknown", "Shadows", "Safe House"],
            NPCRole.HEALER: ["Infirmary", "Herb Shop", "Temple"],
        }

        location_options = locations.get(role, ["Town Square", "Streets", "Unknown"])
        return random.choice(location_options)

    def _generate_faction(self, role: NPCRole) -> str:
        """Generate a faction affiliation for the NPC."""
        factions = {
            NPCRole.NOBLE: ["House Blackstone", "Royal Court", "Merchant Lords"],
            NPCRole.GUARD: ["City Watch", "Royal Guard", "Militia"],
            NPCRole.CRIMINAL: ["Thieves' Guild", "Black Hand", "Shadow Syndicate"],
            NPCRole.SOLDIER: ["King's Army", "Mercenary Company", "City Guard"],
            NPCRole.MAGE: ["Arcane Order", "College of Magic", "Circle of Magi"],
            NPCRole.PRIEST: ["Church of Light", "Temple of the Dawn", "Order of Healing"],
            NPCRole.ASSASSIN: ["Assassin's Guild", "Dark Brotherhood", "Silent Blade"],
            NPCRole.ADVENTURER: ["Adventurer's Guild", "Explorer's Society", "Free Company"],
        }

        faction_options = factions.get(role, ["Independent", "Local Guild", "Free Agent"])
        return random.choice(faction_options)
    
    def _generate_enriched_personality_traits(self, role: NPCRole, custom_traits: Optional[List[str]] = None) -> List[PersonalityTrait]:
        """Generate personality traits using enriched content."""
        traits = []
        
        if custom_traits:
            # Use provided custom traits
            for trait_str in custom_traits:
                traits.append(PersonalityTrait(
                    category="custom",
                    trait=trait_str,
                    description="User-defined trait"
                ))
        
        # Get enriched traits
        npc_traits = self.get_random_npc_traits(count=3)
        
        for char_trait in npc_traits:
            # Categorize the trait
            if char_trait.name in ['STRONG', 'AGILE', 'SCARRED', 'WEATHERED', 'MUSCULAR', 'SLENDER']:
                category = "appearance"
            elif char_trait.name in ['INTELLIGENT', 'CUNNING', 'OBSERVANT', 'CLEVER', 'WISE']:
                category = "intellect"
            elif char_trait.name in ['GRIM', 'CHEERFUL', 'CALM', 'ANXIOUS', 'FIERCE']:
                category = "demeanor"
            else:
                category = "personality"
            
            traits.append(PersonalityTrait(
                category=category,
                trait=char_trait.value.replace('_', ' ').title(),
                description=f"Displays {char_trait.value.replace('_', ' ')} characteristics"
            ))
        
        # Add role-specific trait
        role_traits = self.ROLE_PERSONALITY_TRAITS.get(role, {})
        if role_traits:
            for category, trait_list in role_traits.items():
                if trait_list:
                    selected = random.choice(trait_list)
                    traits.append(PersonalityTrait(
                        category=category,
                        trait=selected,
                        description=f"Role-specific trait for {role.value}"
                    ))
                    break
        
        return traits
    
    def _apply_enriched_traits_to_npc(self, npc: NPC) -> None:
        """Apply enriched character traits to NPC stats and features."""
        # Get appropriate traits for the NPC's role
        trait_count = 2 if npc.importance == "Minor" else 3 if npc.importance == "Supporting" else 4
        selected_traits = self.get_random_npc_traits(count=trait_count)
        
        for trait in selected_traits:
            # Apply stat modifiers based on traits
            if trait == CharacterTrait.STRONG or trait == CharacterTrait.MUSCULAR:
                npc.stats.strength += 2
            elif trait == CharacterTrait.AGILE or trait == CharacterTrait.QUICK:
                npc.stats.dexterity += 2
            elif trait == CharacterTrait.TOUGH or trait == CharacterTrait.RESILIENT:
                npc.stats.constitution += 2
                npc.stats.hit_points += 5
            elif trait == CharacterTrait.INTELLIGENT or trait == CharacterTrait.CLEVER:
                npc.stats.intelligence += 2
            elif trait == CharacterTrait.WISE or trait == CharacterTrait.OBSERVANT:
                npc.stats.wisdom += 2
            elif trait == CharacterTrait.CHARISMATIC or trait == CharacterTrait.CHARMING:
                npc.stats.charisma += 2
            elif trait == CharacterTrait.INTIMIDATING:
                npc.stats.strength += 1
                npc.stats.charisma += 1
            
            # Add trait to features
            npc.features.append(f"Trait: {trait.value.replace('_', ' ').title()}")
    
    def _set_enriched_motivations(self, npc: NPC) -> None:
        """Set NPC motivations using enriched content."""
        # Get primary motivation based on role
        primary_motivation = self.get_npc_motivation(npc.role if npc.role else NPCRole.COMMONER)
        
        # Initialize backstory if needed
        if not npc.backstory:
            from .models import Backstory
            npc.backstory = Backstory()
        
        # Set motivation in backstory
        npc.backstory.motivation = f"Driven by {primary_motivation.value.replace('_', ' ')}"
        
        # Add related goals
        if primary_motivation == CharacterMotivation.WEALTH:
            npc.backstory.goals.append("Accumulate riches")
        elif primary_motivation == CharacterMotivation.KNOWLEDGE:
            npc.backstory.goals.append("Uncover hidden truths")
        elif primary_motivation == CharacterMotivation.POWER:
            npc.backstory.goals.append("Gain influence and control")
        elif primary_motivation == CharacterMotivation.PROTECTION:
            npc.backstory.goals.append("Keep loved ones safe")
        
        # Add related fears
        fear_map = {
            CharacterMotivation.WEALTH: CharacterMotivation.POVERTY,
            CharacterMotivation.POWER: CharacterMotivation.POWERLESSNESS,
            CharacterMotivation.KNOWLEDGE: CharacterMotivation.IGNORANCE,
            CharacterMotivation.PROTECTION: CharacterMotivation.LOSS,
        }
        
        if primary_motivation in fear_map:
            fear = fear_map[primary_motivation]
            npc.backstory.fears.append(f"Fear of {fear.value.replace('_', ' ')}")
