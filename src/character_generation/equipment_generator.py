"""Genre-specific equipment generation system for TTRPG characters and NPCs."""

import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

from .models import (
    CharacterClass,
    Equipment,
    NPCRole,
    TTRPGGenre,
)

logger = logging.getLogger(__name__)


class EquipmentQuality(Enum):
    """Quality levels for equipment."""
    
    POOR = "poor"
    COMMON = "common"
    FINE = "fine"
    MASTERWORK = "masterwork"
    MAGICAL = "magical"
    LEGENDARY = "legendary"
    ARTIFACT = "artifact"


class TechLevel(Enum):
    """Technology levels for sci-fi and modern settings."""
    
    PRIMITIVE = 0
    LOW_TECH = 1
    STANDARD = 2
    HIGH_TECH = 3
    ADVANCED = 4
    EXPERIMENTAL = 5
    ALIEN = 6


@dataclass
class EquipmentItem:
    """Detailed equipment item with genre-specific properties."""
    
    name: str
    item_type: str  # weapon, armor, tool, consumable, etc.
    genre: TTRPGGenre
    description: Optional[str] = None
    quality: EquipmentQuality = EquipmentQuality.COMMON
    value: Optional[int] = None
    weight: Optional[float] = None
    properties: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    tech_level: Optional[TechLevel] = None
    magical: bool = False
    cursed: bool = False
    attunement: bool = False
    charges: Optional[int] = None
    damage: Optional[str] = None
    armor_class: Optional[int] = None
    
    def get_display_name(self) -> str:
        """Get formatted display name with quality."""
        if self.quality == EquipmentQuality.POOR:
            return f"Poor {self.name}"
        elif self.quality == EquipmentQuality.FINE:
            return f"Fine {self.name}"
        elif self.quality == EquipmentQuality.MASTERWORK:
            return f"Masterwork {self.name}"
        elif self.quality == EquipmentQuality.MAGICAL:
            return f"{self.name} +1" if "+" not in self.name else self.name
        elif self.quality == EquipmentQuality.LEGENDARY:
            return f"Legendary {self.name}"
        elif self.quality == EquipmentQuality.ARTIFACT:
            return f"{self.name} (Artifact)"
        return self.name


class EquipmentGenerator:
    """Generate genre-appropriate equipment for TTRPG characters."""
    
    # Fantasy equipment pools
    FANTASY_WEAPONS = {
        "melee": ["Longsword", "Shortsword", "Greatsword", "Battleaxe", "Warhammer",
                  "Mace", "Flail", "Morningstar", "Spear", "Halberd", "Glaive"],
        "ranged": ["Longbow", "Shortbow", "Crossbow", "Heavy Crossbow", "Sling"],
        "light": ["Dagger", "Shortsword", "Rapier", "Scimitar", "Handaxe"],
        "simple": ["Club", "Quarterstaff", "Spear", "Dagger", "Javelin"],
        "exotic": ["Whip", "Net", "Blowgun", "Trident", "War Pick"]
    }
    
    FANTASY_ARMOR = {
        "light": ["Padded Armor", "Leather Armor", "Studded Leather"],
        "medium": ["Hide Armor", "Chain Shirt", "Scale Mail", "Breastplate", "Half Plate"],
        "heavy": ["Ring Mail", "Chain Mail", "Splint Mail", "Plate Mail", "Full Plate"],
        "shields": ["Buckler", "Light Shield", "Heavy Shield", "Tower Shield"]
    }
    
    FANTASY_ITEMS = {
        "adventuring": ["Rope (50 ft)", "Grappling Hook", "Torches", "Lantern", 
                       "Oil Flask", "Tinderbox", "Bedroll", "Rations", "Waterskin"],
        "tools": ["Thieves' Tools", "Healer's Kit", "Herbalism Kit", "Alchemist's Supplies",
                 "Smith's Tools", "Carpenter's Tools", "Mason's Tools"],
        "magical": ["Potion of Healing", "Scroll of Magic Missile", "Wand of Magic Detection",
                   "Ring of Protection", "Cloak of Elvenkind", "Boots of Speed"],
        "consumables": ["Healing Potion", "Antitoxin", "Acid Flask", "Alchemist's Fire",
                       "Holy Water", "Tanglefoot Bag", "Thunderstone"]
    }
    
    # Sci-Fi equipment pools
    SCIFI_WEAPONS = {
        "energy": ["Laser Rifle", "Plasma Pistol", "Ion Cannon", "Pulse Rifle", 
                   "Photon Blade", "Disruptor", "Particle Beam", "Fusion Lance"],
        "kinetic": ["Gauss Rifle", "Rail Gun", "Mass Driver", "Needle Pistol",
                   "Flechette Gun", "Mag-Rifle", "Coil Gun"],
        "melee": ["Vibroblade", "Plasma Sword", "Shock Baton", "Monofilament Whip",
                 "Force Pike", "Energy Staff", "Quantum Blade"],
        "heavy": ["Rocket Launcher", "Grenade Launcher", "Plasma Cannon", "Antimatter Rifle"],
        "special": ["Stun Gun", "Net Launcher", "Sonic Disruptor", "EMP Grenade"]
    }
    
    SCIFI_ARMOR = {
        "light": ["Flex Suit", "Nano-weave Vest", "Energy Shield Belt", "Ablative Coating"],
        "medium": ["Combat Armor", "Powered Exoskeleton", "Reactive Plating", "Ceramic Plates"],
        "heavy": ["Power Armor", "Battle Suit", "Heavy Exo-frame", "Assault Armor"],
        "shields": ["Energy Shield", "Kinetic Barrier", "Deflector Field", "Force Screen"]
    }
    
    SCIFI_ITEMS = {
        "tools": ["Universal Translator", "Med-Scanner", "Quantum Scanner", "Hacking Module",
                 "Repair Kit", "Science Kit", "Navigation Computer", "Comm Unit"],
        "medical": ["Med-Kit", "Stim Pack", "Nano-Medics", "Regen Injector", "Cryo-Stabilizer"],
        "utility": ["Grav-Boots", "Jet Pack", "Cloaking Device", "Holo-Projector",
                   "Force Field Generator", "Portable Shelter", "Survival Kit"],
        "consumables": ["Energy Cell", "Fusion Battery", "Nutrient Paste", "Oxygen Canister"]
    }
    
    # Cyberpunk equipment pools
    CYBERPUNK_WEAPONS = {
        "firearms": ["Heavy Pistol", "SMG", "Assault Rifle", "Shotgun", "Sniper Rifle"],
        "smart": ["Smart Pistol", "Smart Rifle", "Guided Rounds", "Tracking Bullets"],
        "melee": ["Monoblade", "Mantis Blades", "Gorilla Arms", "Nanowire", "Combat Knife"],
        "tech": ["EMP Grenade", "Flashbang", "Smoke Grenade", "Incendiary Grenade"],
        "exotic": ["Rail Gun", "Thermal Katana", "Monowhip", "Tech Shotgun"]
    }
    
    CYBERPUNK_ARMOR = {
        "clothing": ["Armored Jacket", "Bulletproof Vest", "Reinforced Clothing"],
        "armor": ["Light Body Armor", "Medium Body Armor", "Heavy Combat Armor"],
        "subdermal": ["Subdermal Armor", "Skinweave", "Bone Lacing", "Dermal Plating"],
        "shields": ["Reflex Booster", "Sandevistan", "Kerenzikov", "Synaptic Accelerator"]
    }
    
    CYBERPUNK_ITEMS = {
        "cyberware": ["Cybereye", "Neural Interface", "Cyberdeck", "Chipware Socket",
                     "Reflex Booster", "Muscle Enhancement", "Neural Processor"],
        "tech": ["Hacking Deck", "Breaching Protocol", "Scanner", "Jammer", "Bug Detector"],
        "drugs": ["Stims", "Glitter", "Boost", "Synthcoke", "Dorph", "Black Lace"],
        "utility": ["Grappling Gun", "Lock Decoder", "Med-Hypo", "Trauma Kit", "Credchip"]
    }
    
    # Post-Apocalyptic equipment pools
    POST_APOC_WEAPONS = {
        "firearms": ["Pipe Rifle", "Sawed-off Shotgun", "Hunting Rifle", "Revolver",
                    "Makeshift SMG", "Scrap Pistol", "Jury-rigged Assault Rifle"],
        "melee": ["Baseball Bat", "Tire Iron", "Machete", "Fire Axe", "Sledgehammer",
                 "Sharpened Rebar", "Chain", "Nail Board", "Power Fist"],
        "explosive": ["Molotov Cocktail", "Pipe Bomb", "Dynamite", "Grenade", "Mine"],
        "special": ["Flamethrower", "Harpoon Gun", "Crossbow", "Compound Bow"],
        "energy": ["Laser Pistol", "Plasma Rifle", "Gauss Rifle", "Tesla Cannon"]
    }
    
    POST_APOC_ARMOR = {
        "light": ["Leather Jacket", "Padded Clothing", "Road Leathers", "Vault Suit"],
        "medium": ["Metal Armor", "Combat Armor", "Riot Gear", "Raider Armor"],
        "heavy": ["Power Armor", "T-51b Armor", "X-01 Armor", "Salvaged Power Armor"],
        "makeshift": ["Tire Armor", "Scrap Metal Plates", "Sports Padding", "Mutant Hide"]
    }
    
    POST_APOC_ITEMS = {
        "survival": ["Gas Mask", "Rad-Away", "Rad-X", "Stimpak", "Water Purifier",
                    "Geiger Counter", "Hazmat Suit", "Duct Tape", "Scrap Metal"],
        "food": ["Canned Food", "Purified Water", "Brahmin Jerky", "Mutfruit", "Nuka-Cola"],
        "chems": ["Med-X", "Psycho", "Jet", "Buffout", "Mentats", "Fixer"],
        "utility": ["Lockpick Set", "Rope", "Flare", "Compass", "Map", "Binoculars"]
    }
    
    # Western equipment pools
    WESTERN_WEAPONS = {
        "pistols": ["Colt Peacemaker", "Smith & Wesson", "Derringer", "Navy Revolver"],
        "rifles": ["Winchester Rifle", "Henry Rifle", "Sharps Rifle", "Spencer Carbine"],
        "shotguns": ["Double-barrel Shotgun", "Coach Gun", "Sawed-off Shotgun"],
        "melee": ["Bowie Knife", "Tomahawk", "Cavalry Saber", "Bullwhip"],
        "special": ["Dynamite", "Throwing Knife", "Lasso", "Gatling Gun"]
    }
    
    WESTERN_ARMOR = {
        "clothing": ["Duster Coat", "Leather Vest", "Poncho", "Chaps"],
        "armor": ["Boiled Leather", "Steel Breastplate", "Chain Vest"],
        "accessories": ["Gun Belt", "Bandolier", "Holster", "Spurs"]
    }
    
    WESTERN_ITEMS = {
        "gear": ["Saddle", "Saddlebags", "Bedroll", "Canteen", "Compass", "Map"],
        "tools": ["Lock Picks", "Playing Cards", "Dice", "Harmonica", "Gold Pan"],
        "consumables": ["Whiskey", "Tobacco", "Hardtack", "Jerky", "Medicine"],
        "horse": ["Horse", "Mule", "Feed", "Horse Medicine", "Horseshoes"]
    }
    
    # Superhero equipment pools
    SUPERHERO_WEAPONS = {
        "tech": ["Energy Blaster", "Stun Baton", "Web Shooters", "Freeze Ray", "Sonic Cannon"],
        "melee": ["Vibranium Shield", "Adamantium Claws", "Energy Sword", "Power Gauntlets"],
        "gadgets": ["Grappling Hook", "Smoke Bombs", "Flash Grenades", "Tracking Devices"],
        "vehicles": ["Flying Car", "Jet Pack", "Motorcycle", "Submarine", "Spaceship"]
    }
    
    SUPERHERO_ARMOR = {
        "suits": ["Kevlar Suit", "Nano-Suit", "Power Armor", "Vibranium Weave"],
        "accessories": ["Utility Belt", "Cape", "Mask", "Goggles", "Comm Device"],
        "shields": ["Force Field", "Energy Shield", "Kinetic Dampener", "Psi-Shield"]
    }
    
    SUPERHERO_ITEMS = {
        "gadgets": ["Holo-Projector", "Universal Translator", "DNA Scanner", "EMP Device"],
        "medical": ["Healing Factor Serum", "Super Soldier Serum", "Antidote", "Stim Pack"],
        "utility": ["Secret Identity Kit", "Satellite Uplink", "AI Assistant", "Base Computer"]
    }
    
    # Quality modifiers
    QUALITY_MODIFIERS = {
        EquipmentQuality.POOR: {"value": 0.5, "effectiveness": 0.8},
        EquipmentQuality.COMMON: {"value": 1.0, "effectiveness": 1.0},
        EquipmentQuality.FINE: {"value": 2.0, "effectiveness": 1.1},
        EquipmentQuality.MASTERWORK: {"value": 5.0, "effectiveness": 1.2},
        EquipmentQuality.MAGICAL: {"value": 10.0, "effectiveness": 1.5},
        EquipmentQuality.LEGENDARY: {"value": 50.0, "effectiveness": 2.0},
        EquipmentQuality.ARTIFACT: {"value": 100.0, "effectiveness": 3.0}
    }
    
    @classmethod
    def generate_equipment(
        cls,
        genre: TTRPGGenre = TTRPGGenre.FANTASY,
        character_class: Optional[CharacterClass] = None,
        npc_role: Optional[NPCRole] = None,
        level: int = 1,
        wealth_level: str = "standard",  # poor, standard, wealthy, noble
        include_magical: bool = False,
        tech_level: Optional[TechLevel] = None
    ) -> Equipment:
        """
        Generate a complete equipment set for a character.
        
        Args:
            genre: The TTRPG genre for equipment generation
            character_class: Character class for class-specific equipment
            npc_role: NPC role for role-specific equipment
            level: Character level for scaling equipment
            wealth_level: Economic status affecting equipment quality
            include_magical: Whether to include magical/special items
            tech_level: Technology level for sci-fi settings
            
        Returns:
            Complete Equipment object
        """
        equipment = Equipment()
        
        # Generate weapons
        weapons = cls._generate_weapons(genre, character_class, npc_role, level, wealth_level, tech_level)
        equipment.weapons = [w.get_display_name() for w in weapons]
        
        # Generate armor
        armor = cls._generate_armor(genre, character_class, npc_role, level, wealth_level, tech_level)
        equipment.armor = [a.get_display_name() for a in armor]
        
        # Generate items
        items = cls._generate_items(genre, character_class, npc_role, level, wealth_level, tech_level)
        equipment.items = [i.get_display_name() for i in items]
        
        # Generate magical items if appropriate
        if include_magical and level >= 3:
            magic_items = cls._generate_magical_items(genre, level)
            equipment.magic_items = [m.get_display_name() for m in magic_items]
        
        # Generate currency
        equipment.currency = cls._generate_currency(genre, wealth_level, level)
        
        return equipment
    
    @classmethod
    def generate_item(
        cls,
        item_type: str,
        genre: TTRPGGenre = TTRPGGenre.FANTASY,
        quality: Optional[EquipmentQuality] = None,
        magical: bool = False
    ) -> EquipmentItem:
        """Generate a single item of specified type."""
        if quality is None:
            quality = cls._determine_quality(level=1, wealth_level="standard")
        
        # Get appropriate item pool
        if item_type == "weapon":
            item_pool = cls._get_weapon_pool(genre)
        elif item_type == "armor":
            item_pool = cls._get_armor_pool(genre)
        else:
            item_pool = cls._get_item_pool(genre)
        
        # Select random item from pool, filtering for non-empty categories
        valid_categories = [k for k, v in item_pool.items() if v]
        if not valid_categories:
            logger.warning(f"No item categories with items found for genre {genre.value}")
            return EquipmentItem(name="Default Item", item_type=item_type, genre=genre, quality=quality, magical=magical)
        category = random.choice(valid_categories)
        name = random.choice(item_pool[category])
        
        # Create item
        item = EquipmentItem(
            name=name,
            item_type=item_type,
            genre=genre,
            quality=quality,
            magical=magical
        )
        
        # Add properties based on type and genre
        cls._add_item_properties(item)
        
        # Add magical properties if needed
        if magical:
            cls._add_magical_properties(item)
        
        return item
    
    @classmethod
    def _generate_weapons(
        cls,
        genre: TTRPGGenre,
        character_class: Optional[CharacterClass],
        npc_role: Optional[NPCRole],
        level: int,
        wealth_level: str,
        tech_level: Optional[TechLevel] = None
    ) -> List[EquipmentItem]:
        """Generate weapons based on context."""
        weapons = []
        weapon_pool = cls._get_weapon_pool(genre)
        
        # Determine number of weapons
        num_weapons = 1
        if level >= 5 or wealth_level in ["wealthy", "noble"]:
            num_weapons = 2
        if npc_role in [NPCRole.GUARD, NPCRole.SOLDIER, NPCRole.ADVENTURER]:
            num_weapons = 2
        
        # Generate primary weapon
        primary_category = cls._get_primary_weapon_category(genre, character_class, npc_role)
        if primary_category and primary_category in weapon_pool:
            weapon_name = random.choice(weapon_pool[primary_category])
            quality = cls._determine_quality(level, wealth_level)
            
            weapon = EquipmentItem(
                name=weapon_name,
                item_type="weapon",
                genre=genre,
                quality=quality
            )
            cls._add_weapon_properties(weapon, genre, primary_category, tech_level)
            weapons.append(weapon)
        
        # Generate secondary weapon if needed
        if num_weapons > 1:
            secondary_category = cls._get_secondary_weapon_category(genre)
            if secondary_category and secondary_category in weapon_pool:
                weapon_name = random.choice(weapon_pool[secondary_category])
                quality = cls._determine_quality(level - 2, wealth_level)
                
                weapon = EquipmentItem(
                    name=weapon_name,
                    item_type="weapon",
                    genre=genre,
                    quality=quality
                )
                cls._add_weapon_properties(weapon, genre, secondary_category, tech_level)
                weapons.append(weapon)
        
        return weapons
    
    @classmethod
    def _generate_armor(
        cls,
        genre: TTRPGGenre,
        character_class: Optional[CharacterClass],
        npc_role: Optional[NPCRole],
        level: int,
        wealth_level: str,
        tech_level: Optional[TechLevel] = None
    ) -> List[EquipmentItem]:
        """Generate armor based on context."""
        armor_items = []
        armor_pool = cls._get_armor_pool(genre)
        
        # Determine armor category
        armor_category = cls._get_armor_category(genre, character_class, npc_role)
        
        if armor_category and armor_category in armor_pool:
            armor_name = random.choice(armor_pool[armor_category])
            quality = cls._determine_quality(level, wealth_level)
            
            armor = EquipmentItem(
                name=armor_name,
                item_type="armor",
                genre=genre,
                quality=quality
            )
            cls._add_armor_properties(armor, genre, armor_category, tech_level)
            armor_items.append(armor)
        
        # Add shield if appropriate
        if cls._should_have_shield(character_class, npc_role):
            if "shields" in armor_pool:
                shield_name = random.choice(armor_pool["shields"])
                shield = EquipmentItem(
                    name=shield_name,
                    item_type="shield",
                    genre=genre,
                    quality=cls._determine_quality(level - 1, wealth_level)
                )
                armor_items.append(shield)
        
        return armor_items
    
    @classmethod
    def _generate_items(
        cls,
        genre: TTRPGGenre,
        character_class: Optional[CharacterClass],
        npc_role: Optional[NPCRole],
        level: int,
        wealth_level: str,
        tech_level: Optional[TechLevel] = None
    ) -> List[EquipmentItem]:
        """Generate miscellaneous items based on context."""
        items = []
        item_pool = cls._get_item_pool(genre)
        
        # Determine number of items
        num_items = 3 + (level // 3)
        if wealth_level in ["wealthy", "noble"]:
            num_items += 2
        
        # Get relevant categories
        categories = cls._get_relevant_item_categories(genre, character_class, npc_role)
        
        for _ in range(num_items):
            if categories and random.random() < 0.7:  # 70% chance for relevant items
                category = random.choice(categories)
            else:
                category = random.choice(list(item_pool.keys()))
            
            if category in item_pool and item_pool[category]:
                item_name = random.choice(item_pool[category])
                quality = cls._determine_quality(max(1, level - 2), wealth_level)
                
                item = EquipmentItem(
                    name=item_name,
                    item_type="item",
                    genre=genre,
                    quality=quality if "consumable" not in category else EquipmentQuality.COMMON
                )
                items.append(item)
        
        return items
    
    @classmethod
    def _generate_magical_items(
        cls,
        genre: TTRPGGenre,
        level: int
    ) -> List[EquipmentItem]:
        """Generate magical or special items based on level."""
        magical_items = []
        
        # Determine number of magical items
        num_magical = 0
        if 3 <= level < 5:
            num_magical = 1
        elif 5 <= level < 10:
            num_magical = random.randint(1, 2)
        elif level >= 10:
            num_magical = random.randint(2, 3)
        
        for _ in range(num_magical):
            item_type = random.choice(["weapon", "armor", "accessory", "consumable"])
            
            if genre == TTRPGGenre.FANTASY:
                item = cls._generate_fantasy_magical_item(item_type, level)
            elif genre in [TTRPGGenre.SCI_FI, TTRPGGenre.CYBERPUNK]:
                item = cls._generate_tech_special_item(item_type, level)
            else:
                item = cls._generate_special_item(genre, item_type, level)
            
            magical_items.append(item)
        
        return magical_items
    
    @classmethod
    def _generate_currency(
        cls,
        genre: TTRPGGenre,
        wealth_level: str,
        level: int
    ) -> Dict[str, int]:
        """Generate appropriate currency for the character."""
        currency = {}
        
        # Base values by wealth level
        base_wealth = {
            "poor": 10,
            "standard": 50,
            "wealthy": 200,
            "noble": 1000
        }
        
        base = base_wealth.get(wealth_level, 50)
        multiplier = 1 + (level * 0.5)
        
        if genre == TTRPGGenre.FANTASY:
            total_value = int(base * multiplier)
            currency["gold"] = total_value // 10
            currency["silver"] = (total_value % 10) * 10
            currency["copper"] = random.randint(0, 100)
        elif genre in [TTRPGGenre.SCI_FI, TTRPGGenre.CYBERPUNK]:
            currency["credits"] = int(base * multiplier * 10)
        elif genre == TTRPGGenre.POST_APOCALYPTIC:
            currency["caps"] = int(base * multiplier)
            currency["scrap"] = random.randint(5, 50)
        elif genre == TTRPGGenre.WESTERN:
            currency["dollars"] = int(base * multiplier)
            currency["cents"] = random.randint(0, 99)
        else:
            currency["money"] = int(base * multiplier)
        
        return currency
    
    @classmethod
    def _get_weapon_pool(cls, genre: TTRPGGenre) -> Dict[str, List[str]]:
        """Get the appropriate weapon pool for the genre."""
        pools = {
            TTRPGGenre.FANTASY: cls.FANTASY_WEAPONS,
            TTRPGGenre.SCI_FI: cls.SCIFI_WEAPONS,
            TTRPGGenre.CYBERPUNK: cls.CYBERPUNK_WEAPONS,
            TTRPGGenre.POST_APOCALYPTIC: cls.POST_APOC_WEAPONS,
            TTRPGGenre.WESTERN: cls.WESTERN_WEAPONS,
            TTRPGGenre.SUPERHERO: cls.SUPERHERO_WEAPONS,
            TTRPGGenre.COSMIC_HORROR: cls.WESTERN_WEAPONS,  # Early 20th century
            TTRPGGenre.STEAMPUNK: cls.FANTASY_WEAPONS,  # With modifications
        }
        return pools.get(genre, cls.FANTASY_WEAPONS)
    
    @classmethod
    def _get_armor_pool(cls, genre: TTRPGGenre) -> Dict[str, List[str]]:
        """Get the appropriate armor pool for the genre."""
        pools = {
            TTRPGGenre.FANTASY: cls.FANTASY_ARMOR,
            TTRPGGenre.SCI_FI: cls.SCIFI_ARMOR,
            TTRPGGenre.CYBERPUNK: cls.CYBERPUNK_ARMOR,
            TTRPGGenre.POST_APOCALYPTIC: cls.POST_APOC_ARMOR,
            TTRPGGenre.WESTERN: cls.WESTERN_ARMOR,
            TTRPGGenre.SUPERHERO: cls.SUPERHERO_ARMOR,
            TTRPGGenre.COSMIC_HORROR: cls.WESTERN_ARMOR,
            TTRPGGenre.STEAMPUNK: cls.FANTASY_ARMOR,
        }
        return pools.get(genre, cls.FANTASY_ARMOR)
    
    @classmethod
    def _get_item_pool(cls, genre: TTRPGGenre) -> Dict[str, List[str]]:
        """Get the appropriate item pool for the genre."""
        pools = {
            TTRPGGenre.FANTASY: cls.FANTASY_ITEMS,
            TTRPGGenre.SCI_FI: cls.SCIFI_ITEMS,
            TTRPGGenre.CYBERPUNK: cls.CYBERPUNK_ITEMS,
            TTRPGGenre.POST_APOCALYPTIC: cls.POST_APOC_ITEMS,
            TTRPGGenre.WESTERN: cls.WESTERN_ITEMS,
            TTRPGGenre.SUPERHERO: cls.SUPERHERO_ITEMS,
            TTRPGGenre.COSMIC_HORROR: cls.WESTERN_ITEMS,
            TTRPGGenre.STEAMPUNK: cls.FANTASY_ITEMS,
        }
        return pools.get(genre, cls.FANTASY_ITEMS)
    
    @classmethod
    def _determine_quality(cls, level: int, wealth_level: str) -> EquipmentQuality:
        """Determine equipment quality based on level and wealth."""
        # Base chances
        quality_chances = {
            "poor": {
                EquipmentQuality.POOR: 0.4,
                EquipmentQuality.COMMON: 0.5,
                EquipmentQuality.FINE: 0.1
            },
            "standard": {
                EquipmentQuality.POOR: 0.1,
                EquipmentQuality.COMMON: 0.6,
                EquipmentQuality.FINE: 0.25,
                EquipmentQuality.MASTERWORK: 0.05
            },
            "wealthy": {
                EquipmentQuality.COMMON: 0.3,
                EquipmentQuality.FINE: 0.5,
                EquipmentQuality.MASTERWORK: 0.15,
                EquipmentQuality.MAGICAL: 0.05
            },
            "noble": {
                EquipmentQuality.FINE: 0.3,
                EquipmentQuality.MASTERWORK: 0.5,
                EquipmentQuality.MAGICAL: 0.15,
                EquipmentQuality.LEGENDARY: 0.05
            }
        }
        
        chances = quality_chances.get(wealth_level, quality_chances["standard"]).copy()
        
        # Adjust for level
        if level >= 10:
            # Shift probabilities toward better quality
            if EquipmentQuality.MAGICAL in chances:
                chances[EquipmentQuality.MAGICAL] *= 2
            if EquipmentQuality.LEGENDARY not in chances:
                chances[EquipmentQuality.LEGENDARY] = 0.02
            
            # Re-normalize probabilities to ensure they sum to 1.0
            total_chance = sum(chances.values())
            if total_chance > 0:
                chances = {q: p / total_chance for q, p in chances.items()}
        
        # Select quality based on weighted random
        roll = random.random()
        cumulative = 0
        
        for quality, chance in chances.items():
            cumulative += chance
            if roll < cumulative:
                return quality
        
        return EquipmentQuality.COMMON
    
    # Data-driven weapon category mappings
    WEAPON_CATEGORY_MAP = {
        TTRPGGenre.FANTASY: {
            CharacterClass.FIGHTER: "melee",
            CharacterClass.PALADIN: "melee",
            CharacterClass.RANGER: ["light", "ranged"],
            CharacterClass.ROGUE: ["light", "ranged"],
            CharacterClass.WIZARD: "simple",
            CharacterClass.SORCERER: "simple",
            NPCRole.GUARD: "melee",
            NPCRole.CRIMINAL: "light",
            "default": ["melee", "light", "ranged", "simple"]
        },
        TTRPGGenre.SCI_FI: {
            CharacterClass.MARINE: "kinetic",
            CharacterClass.TECH_SPECIALIST: "energy",
            "default": ["kinetic", "energy"]
        },
        TTRPGGenre.CYBERPUNK: {
            NPCRole.CRIMINAL: "firearms",
            "default": ["firearms", "smart"]
        },
        TTRPGGenre.POST_APOCALYPTIC: {
            "default": ["firearms", "melee"]
        },
        TTRPGGenre.WESTERN: {
            "default": ["pistols", "rifles"]
        }
    }

    @classmethod
    def _get_primary_weapon_category(
        cls,
        genre: TTRPGGenre,
        character_class: Optional[CharacterClass],
        npc_role: Optional[NPCRole]
    ) -> Optional[str]:
        """Determine primary weapon category based on class/role using data-driven approach."""
        if genre not in cls.WEAPON_CATEGORY_MAP:
            # Fallback to first available category from pool
            weapon_pool = cls._get_weapon_pool(genre)
            return list(weapon_pool.keys())[0] if weapon_pool else None
        
        genre_map = cls.WEAPON_CATEGORY_MAP[genre]
        
        # Check character class first
        if character_class and character_class in genre_map:
            category = genre_map[character_class]
            return random.choice(category) if isinstance(category, list) else category
        
        # Check NPC role next
        if npc_role and npc_role in genre_map:
            category = genre_map[npc_role]
            return random.choice(category) if isinstance(category, list) else category
        
        # Use default for genre
        default_categories = genre_map.get("default", [])
        if default_categories:
            return random.choice(default_categories)
        
        # Final fallback
        weapon_pool = cls._get_weapon_pool(genre)
        return list(weapon_pool.keys())[0] if weapon_pool else None
    
    # Data-driven secondary weapon category mappings
    SECONDARY_WEAPON_MAP = {
        TTRPGGenre.FANTASY: "light",
        TTRPGGenre.SCI_FI: "melee", 
        TTRPGGenre.CYBERPUNK: "melee",
        TTRPGGenre.POST_APOCALYPTIC: ["melee", "explosive"],  # weighted choice
        TTRPGGenre.WESTERN: "melee"
    }

    @classmethod
    def _get_secondary_weapon_category(cls, genre: TTRPGGenre) -> Optional[str]:
        """Determine secondary weapon category using data-driven approach."""
        if genre not in cls.SECONDARY_WEAPON_MAP:
            return None
        
        category = cls.SECONDARY_WEAPON_MAP[genre]
        if isinstance(category, list):
            # For POST_APOCALYPTIC, use weighted choice (70% melee, 30% explosive)
            if genre == TTRPGGenre.POST_APOCALYPTIC:
                return "melee" if random.random() < 0.7 else "explosive"
            else:
                return random.choice(category)
        
        return category
    
    # Data-driven armor category mappings
    ARMOR_CATEGORY_MAP = {
        TTRPGGenre.FANTASY: {
            CharacterClass.FIGHTER: "heavy",
            CharacterClass.PALADIN: "heavy",
            CharacterClass.RANGER: "light",
            CharacterClass.ROGUE: "light",
            CharacterClass.WIZARD: None,  # No armor
            CharacterClass.SORCERER: None,  # No armor
            NPCRole.GUARD: "medium",
            NPCRole.NOBLE: "light",
            "default": "light"
        },
        TTRPGGenre.SCI_FI: {
            CharacterClass.MARINE: "heavy",
            "default": "medium"
        },
        TTRPGGenre.CYBERPUNK: {
            NPCRole.CRIMINAL: "clothing",
            "default": "armor"
        },
        TTRPGGenre.POST_APOCALYPTIC: {
            "default": ["light", "medium", "makeshift"]
        },
        TTRPGGenre.WESTERN: {
            "default": "clothing"
        }
    }

    @classmethod
    def _get_armor_category(
        cls,
        genre: TTRPGGenre,
        character_class: Optional[CharacterClass],
        npc_role: Optional[NPCRole]
    ) -> Optional[str]:
        """Determine armor category based on class/role using data-driven approach."""
        if genre not in cls.ARMOR_CATEGORY_MAP:
            return "light"  # Fallback default
        
        genre_map = cls.ARMOR_CATEGORY_MAP[genre]
        
        # Check character class first
        if character_class and character_class in genre_map:
            category = genre_map[character_class]
            return random.choice(category) if isinstance(category, list) else category
        
        # Check NPC role next
        if npc_role and npc_role in genre_map:
            category = genre_map[npc_role]
            return random.choice(category) if isinstance(category, list) else category
        
        # Use default for genre
        default_category = genre_map.get("default", "light")
        return random.choice(default_category) if isinstance(default_category, list) else default_category
    
    @classmethod
    def _should_have_shield(
        cls,
        character_class: Optional[CharacterClass],
        npc_role: Optional[NPCRole]
    ) -> bool:
        """Determine if character should have a shield."""
        shield_classes = [
            CharacterClass.FIGHTER,
            CharacterClass.PALADIN,
            CharacterClass.CLERIC
        ]
        shield_roles = [NPCRole.GUARD, NPCRole.SOLDIER]
        
        if character_class in shield_classes:
            return True
        if npc_role in shield_roles:
            return random.random() < 0.6
        return False
    
    @classmethod
    def _get_relevant_item_categories(
        cls,
        genre: TTRPGGenre,
        character_class: Optional[CharacterClass],
        npc_role: Optional[NPCRole]
    ) -> List[str]:
        """Get item categories relevant to the character."""
        categories = []
        
        if genre == TTRPGGenre.FANTASY:
            if character_class == CharacterClass.ROGUE or npc_role == NPCRole.CRIMINAL:
                categories.append("tools")
            if character_class == CharacterClass.CLERIC or npc_role == NPCRole.PRIEST:
                categories.append("consumables")
            categories.append("adventuring")
        elif genre == TTRPGGenre.SCI_FI:
            categories.extend(["tools", "medical", "utility"])
        elif genre == TTRPGGenre.CYBERPUNK:
            if character_class == CharacterClass.NETRUNNER:
                categories.append("tech")
            categories.extend(["cyberware", "drugs"])
        elif genre == TTRPGGenre.POST_APOCALYPTIC:
            categories.extend(["survival", "food", "chems"])
        elif genre == TTRPGGenre.WESTERN:
            categories.extend(["gear", "consumables"])
            if npc_role == NPCRole.CRIMINAL:
                categories.append("tools")
        
        return categories
    
    @classmethod
    def _add_weapon_properties(
        cls,
        weapon: EquipmentItem,
        genre: TTRPGGenre,
        category: str,
        tech_level: Optional[TechLevel] = None
    ) -> None:
        """Add properties to a weapon based on its type."""
        if genre == TTRPGGenre.FANTASY:
            if category == "melee":
                weapon.damage = f"1d{random.choice([6, 8, 10])} slashing"
                weapon.properties.append("Melee")
            elif category == "ranged":
                weapon.damage = f"1d{random.choice([6, 8])} piercing"
                weapon.properties.extend(["Ranged", f"Range {random.choice([80, 120, 150])}ft"])
            elif category == "light":
                weapon.damage = "1d6 piercing"
                weapon.properties.extend(["Light", "Finesse"])
        elif genre == TTRPGGenre.SCI_FI:
            if category == "energy":
                weapon.damage = f"{random.randint(2, 4)}d6 energy"
                weapon.properties.extend(["Energy", "Rechargeable"])
                weapon.tech_level = tech_level or TechLevel.HIGH_TECH
            elif category == "kinetic":
                weapon.damage = f"{random.randint(2, 3)}d8 kinetic"
                weapon.properties.extend(["Kinetic", "Armor Piercing"])
        elif genre == TTRPGGenre.POST_APOCALYPTIC:
            if category == "makeshift":
                weapon.properties.extend(["Improvised", "Unreliable"])
            weapon.properties.append("Scavenged")
    
    @classmethod
    def _add_armor_properties(
        cls,
        armor: EquipmentItem,
        genre: TTRPGGenre,
        category: str,
        tech_level: Optional[TechLevel] = None
    ) -> None:
        """Add properties to armor based on its type."""
        ac_values = {
            "light": random.randint(11, 13),
            "medium": random.randint(14, 16),
            "heavy": random.randint(17, 19),
            "clothing": random.randint(10, 11),
            "shields": random.randint(1, 3)  # Shield bonus
        }
        
        armor.armor_class = ac_values.get(category, 10)
        
        if category == "heavy":
            armor.properties.append("Heavy")
            armor.requirements.append("Strength 13+")
        elif category == "light":
            armor.properties.append("Light")
        
        if genre == TTRPGGenre.SCI_FI:
            armor.properties.append("Powered")
            armor.tech_level = tech_level or TechLevel.HIGH_TECH
        elif genre == TTRPGGenre.POST_APOCALYPTIC:
            armor.properties.append("Makeshift")
    
    @classmethod
    def _add_item_properties(cls, item: EquipmentItem) -> None:
        """Add general properties to an item."""
        # Add weight
        item.weight = random.uniform(0.1, 5.0)
        
        # Add value based on quality
        base_value = random.randint(1, 100)
        quality_mod = cls.QUALITY_MODIFIERS[item.quality]["value"]
        item.value = int(base_value * quality_mod)
    
    @classmethod
    def _add_magical_properties(cls, item: EquipmentItem) -> None:
        """Add magical properties to an item."""
        item.magical = True
        # Only set quality to MAGICAL if it's not already a higher quality
        if item.quality == EquipmentQuality.COMMON or item.quality == EquipmentQuality.FINE:
            item.quality = EquipmentQuality.MAGICAL
        
        magical_properties = [
            "+1 Enhancement",
            "Flaming",
            "Frost",
            "Shock",
            "Keen",
            "Vorpal",
            "Holy",
            "Unholy",
            "Speed",
            "Defense"
        ]
        
        num_properties = 1
        if item.quality == EquipmentQuality.LEGENDARY:
            num_properties = random.randint(2, 3)
        elif item.quality == EquipmentQuality.ARTIFACT:
            num_properties = random.randint(3, 5)
        
        selected = random.sample(magical_properties, min(num_properties, len(magical_properties)))
        item.properties.extend(selected)
        
        # Add attunement requirement for powerful items
        if item.quality in [EquipmentQuality.LEGENDARY, EquipmentQuality.ARTIFACT]:
            item.attunement = True
    
    @classmethod
    def _generate_fantasy_magical_item(
        cls,
        item_type: str,
        level: int
    ) -> EquipmentItem:
        """Generate a fantasy magical item."""
        magic_items = {
            "weapon": ["Flaming Sword", "Frost Brand", "Holy Avenger", "Vorpal Blade"],
            "armor": ["Mithril Shirt", "Dragon Scale Mail", "Elven Chain", "Adamantine Plate"],
            "accessory": ["Ring of Protection", "Cloak of Elvenkind", "Boots of Speed", "Amulet of Health"],
            "consumable": ["Potion of Healing", "Scroll of Fireball", "Wand of Magic Missiles"]
        }
        
        name = random.choice(magic_items.get(item_type, magic_items["accessory"]))
        quality = EquipmentQuality.MAGICAL if level < 10 else EquipmentQuality.LEGENDARY
        
        item = EquipmentItem(
            name=name,
            item_type=item_type,
            genre=TTRPGGenre.FANTASY,
            quality=quality,
            magical=True
        )
        
        cls._add_magical_properties(item)
        return item
    
    @classmethod
    def _generate_tech_special_item(
        cls,
        item_type: str,
        level: int
    ) -> EquipmentItem:
        """Generate a high-tech special item."""
        tech_items = {
            "weapon": ["Plasma Cannon Mk-V", "Quantum Disruptor", "Neural Shredder"],
            "armor": ["Nanoweave Suit", "Force Field Generator", "Adaptive Camouflage"],
            "accessory": ["Neural Implant", "Quantum Computer", "Teleporter Beacon"],
            "consumable": ["Nano-Repair Kit", "Stim Pack Ultra", "Time Dilation Field"]
        }
        
        name = random.choice(tech_items.get(item_type, tech_items["accessory"]))
        
        item = EquipmentItem(
            name=name,
            item_type=item_type,
            genre=TTRPGGenre.SCI_FI,
            quality=EquipmentQuality.MASTERWORK if level < 10 else EquipmentQuality.LEGENDARY,
            tech_level=TechLevel.ADVANCED if level < 10 else TechLevel.EXPERIMENTAL
        )
        
        item.properties.append("High-Tech")
        return item
    
    @classmethod
    def _generate_special_item(
        cls,
        genre: TTRPGGenre,
        item_type: str,
        level: int
    ) -> EquipmentItem:
        """Generate a genre-appropriate special item."""
        special_items = {
            TTRPGGenre.POST_APOCALYPTIC: {
                "weapon": ["Pre-War Laser Rifle", "Experimental Plasma Caster"],
                "armor": ["Pristine Power Armor", "Vault-Tec Prototype Suit"],
                "accessory": ["Pip-Boy 3000", "G.E.C.K."],
                "consumable": ["Pure Rad-Away", "Super Stimpak"]
            },
            TTRPGGenre.WESTERN: {
                "weapon": ["Blessed Six-Shooter", "Lightning Rod Rifle"],
                "armor": ["Blessed Poncho", "Bulletproof Vest"],
                "accessory": ["Lucky Charm", "Ancient Map"],
                "consumable": ["Snake Oil Elixir", "Dynamite Bundle"]
            },
            TTRPGGenre.SUPERHERO: {
                "weapon": ["Alien Tech Blaster", "Vibranium Weapon"],
                "armor": ["Nano-Suit", "Alien Symbiote"],
                "accessory": ["Power Ring", "Infinity Stone Fragment"],
                "consumable": ["Super Soldier Serum", "Mutation Catalyst"]
            }
        }
        
        # Fallback to fantasy items for genres not explicitly defined
        fallback_items = {
            "weapon": ["Enchanted Weapon", "Blessed Blade", "Ancient Artifact"],
            "armor": ["Enchanted Armor", "Protective Ward", "Ancient Protection"],
            "accessory": ["Mystic Amulet", "Ancient Ring", "Power Crystal"],
            "consumable": ["Healing Elixir", "Power Potion", "Ancient Scroll"]
        }
        genre_items = special_items.get(genre, fallback_items)
        name = random.choice(genre_items.get(item_type, genre_items["accessory"]))
        
        item = EquipmentItem(
            name=name,
            item_type=item_type,
            genre=genre,
            quality=EquipmentQuality.LEGENDARY if level >= 10 else EquipmentQuality.MASTERWORK
        )
        
        item.properties.append("Unique")
        return item