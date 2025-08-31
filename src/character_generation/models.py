"""Data models for character and NPC generation."""

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TTRPGGenre(Enum):
    """TTRPG genre categories."""
    
    FANTASY = "fantasy"
    SCI_FI = "sci-fi"
    CYBERPUNK = "cyberpunk"
    COSMIC_HORROR = "cosmic_horror"
    POST_APOCALYPTIC = "post_apocalyptic"
    SUPERHERO = "superhero"
    STEAMPUNK = "steampunk"
    WESTERN = "western"
    MODERN = "modern"
    SPACE_OPERA = "space_opera"
    URBAN_FANTASY = "urban_fantasy"
    HISTORICAL = "historical"
    CUSTOM = "custom"


class CharacterClass(Enum):
    """Character classes across multiple TTRPG genres."""
    
    # Fantasy Classes (backward compatible)
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
    
    # Sci-Fi Classes
    ENGINEER = "engineer"
    SCIENTIST = "scientist"
    PILOT = "pilot"
    MARINE = "marine"
    DIPLOMAT = "diplomat"
    XENOBIOLOGIST = "xenobiologist"
    TECH_SPECIALIST = "tech_specialist"
    PSION = "psion"
    BOUNTY_HUNTER = "bounty_hunter"
    
    # Cyberpunk Classes
    NETRUNNER = "netrunner"
    SOLO = "solo"
    FIXER = "fixer"
    CORPORATE = "corporate"
    ROCKERBOY = "rockerboy"
    TECHIE = "techie"
    MEDIA = "media"
    COP = "cop"
    NOMAD = "nomad"
    
    # Cosmic Horror Classes
    INVESTIGATOR = "investigator"
    SCHOLAR = "scholar"
    ANTIQUARIAN = "antiquarian"
    OCCULTIST = "occultist"
    ALIENIST = "alienist"
    ARCHAEOLOGIST = "archaeologist"
    JOURNALIST = "journalist"
    DETECTIVE = "detective"
    PROFESSOR = "professor"
    
    # Post-Apocalyptic Classes
    SURVIVOR = "survivor"
    SCAVENGER = "scavenger"
    RAIDER = "raider"
    MEDIC = "medic"
    MECHANIC = "mechanic"
    TRADER = "trader"
    WARLORD = "warlord"
    MUTANT_HUNTER = "mutant_hunter"
    VAULT_DWELLER = "vault_dweller"
    
    # Superhero Classes
    VIGILANTE = "vigilante"
    POWERED = "powered"
    GENIUS = "genius"
    MARTIAL_ARTIST = "martial_artist"
    MYSTIC = "mystic"
    ALIEN_HERO = "alien_hero"
    TECH_HERO = "tech_hero"
    SIDEKICK = "sidekick"
    
    # Western Classes
    GUNSLINGER = "gunslinger"
    LAWMAN = "lawman"
    OUTLAW = "outlaw"
    GAMBLER = "gambler"
    PREACHER = "preacher"
    PROSPECTOR = "prospector"
    NATIVE_SCOUT = "native_scout"
    
    CUSTOM = "custom"


class CharacterRace(Enum):
    """Character races across multiple TTRPG genres."""
    
    # Fantasy Races (backward compatible)
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
    
    # Sci-Fi Races
    TERRAN = "terran"
    MARTIAN = "martian"
    BELTER = "belter"
    CYBORG = "cyborg"
    ANDROID = "android"
    AI_CONSTRUCT = "ai_construct"
    GREY_ALIEN = "grey_alien"
    REPTILIAN = "reptilian"
    INSECTOID = "insectoid"
    ENERGY_BEING = "energy_being"
    SILICON_BASED = "silicon_based"
    UPLIFTED_ANIMAL = "uplifted_animal"
    
    # Cyberpunk Races
    AUGMENTED_HUMAN = "augmented_human"
    FULL_CONVERSION_CYBORG = "full_conversion_cyborg"
    BIOENGINEERED = "bioengineered"
    CLONE = "clone"
    DIGITAL_CONSCIOUSNESS = "digital_consciousness"
    
    # Cosmic Horror Races
    DEEP_ONE_HYBRID = "deep_one_hybrid"
    GHOUL = "ghoul"
    DREAMLANDS_NATIVE = "dreamlands_native"
    TOUCHED = "touched"
    
    # Post-Apocalyptic Races
    PURE_STRAIN_HUMAN = "pure_strain_human"
    MUTANT = "mutant"
    GHOUL_WASTELANDER = "ghoul_wastelander"
    SYNTHETIC = "synthetic"
    HYBRID = "hybrid"
    RADIANT = "radiant"
    
    # Superhero Races
    METAHUMAN = "metahuman"
    INHUMAN = "inhuman"
    ATLANTEAN = "atlantean"
    AMAZONIAN = "amazonian"
    KRYPTONIAN = "kryptonian"
    ASGARDIAN = "asgardian"
    
    CUSTOM = "custom"


class NPCRole(Enum):
    """NPC roles across multiple TTRPG genres."""
    
    # Fantasy Roles (backward compatible)
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
    
    # Sci-Fi Roles
    STATION_COMMANDER = "station_commander"
    SHIP_CAPTAIN = "ship_captain"
    COLONY_ADMINISTRATOR = "colony_administrator"
    XENOBIOLOGIST_NPC = "xenobiologist_npc"
    SPACE_TRADER = "space_trader"
    ASTEROID_MINER = "asteroid_miner"
    JUMP_GATE_OPERATOR = "jump_gate_operator"
    ALIEN_AMBASSADOR = "alien_ambassador"
    SMUGGLER = "smuggler"
    
    # Cyberpunk Roles
    STREET_SAMURAI = "street_samurai"
    CORPORATE_EXEC = "corporate_exec"
    BLACK_MARKET_DEALER = "black_market_dealer"
    RIPPERDOC = "ripperdoc"
    GANG_LEADER = "gang_leader"
    INFO_BROKER = "info_broker"
    CLUB_OWNER = "club_owner"
    CORRUPT_COP = "corrupt_cop"
    
    # Cosmic Horror Roles
    CULTIST = "cultist"
    CULT_LEADER = "cult_leader"
    MAD_SCIENTIST = "mad_scientist"
    LIBRARIAN = "librarian"
    ASYLUM_DOCTOR = "asylum_doctor"
    PRIVATE_INVESTIGATOR = "private_investigator"
    MUSEUM_CURATOR = "museum_curator"
    DOOMSDAY_PROPHET = "doomsday_prophet"
    
    # Post-Apocalyptic Roles
    SETTLEMENT_LEADER = "settlement_leader"
    WASTELAND_DOCTOR = "wasteland_doctor"
    CARAVAN_MASTER = "caravan_master"
    RAIDER_CHIEF = "raider_chief"
    VAULT_OVERSEER = "vault_overseer"
    SCRAP_DEALER = "scrap_dealer"
    WATER_MERCHANT = "water_merchant"
    TRIBAL_ELDER = "tribal_elder"
    
    # Superhero Roles
    POLICE_COMMISSIONER = "police_commissioner"
    NEWS_REPORTER = "news_reporter"
    SCIENTIST_ALLY = "scientist_ally"
    GOVERNMENT_AGENT = "government_agent"
    VILLAIN = "villain"
    HENCHMAN = "henchman"
    CIVILIAN = "civilian"
    
    # Western Roles
    SHERIFF = "sheriff"
    SALOON_KEEPER = "saloon_keeper"
    RANCH_OWNER = "ranch_owner"
    BANK_TELLER = "bank_teller"
    STAGE_COACH_DRIVER = "stage_coach_driver"
    BLACKSMITH = "blacksmith"
    MEDICINE_MAN = "medicine_man"
    
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
    def from_dict(cls, data: Dict[str, Any]) -> "CharacterStats":
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
    currency: Dict[str, int] = field(default_factory=lambda: {"gold": 0, "silver": 0, "copper": 0})
    magic_items: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert equipment to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Equipment":
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
    def from_dict(cls, data: Dict[str, Any]) -> "Backstory":
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
    genre: Optional[TTRPGGenre] = TTRPGGenre.FANTASY  # Genre category

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
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        if self.character_class:
            data["character_class"] = self.character_class.value
        if self.race:
            data["race"] = self.race.value
        if self.genre:
            data["genre"] = self.genre.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Character":
        """Create character from dictionary."""
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if isinstance(data.get("updated_at"), str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        # Handle genre field
        genre_str = data.get("genre")
        if isinstance(genre_str, str):
            try:
                data["genre"] = TTRPGGenre(genre_str)
            except ValueError:
                data["genre"] = TTRPGGenre.FANTASY  # Default to fantasy for backward compatibility
        elif "genre" not in data:
            data["genre"] = TTRPGGenre.FANTASY  # Default for backward compatibility

        class_str = data.get("character_class")
        if isinstance(class_str, str):
            try:
                data["character_class"] = CharacterClass(class_str)
            except ValueError:
                data["character_class"] = CharacterClass.CUSTOM
                data["custom_class"] = class_str

        if isinstance(data.get("race"), str):
            try:
                data["race"] = CharacterRace(data["race"])
            except ValueError:
                data["race"] = CharacterRace.CUSTOM
                data["custom_race"] = data.get("race")

        if isinstance(data.get("stats"), dict):
            data["stats"] = CharacterStats.from_dict(data["stats"])
        if isinstance(data.get("equipment"), dict):
            data["equipment"] = Equipment.from_dict(data["equipment"])
        if isinstance(data.get("backstory"), dict):
            data["backstory"] = Backstory.from_dict(data["backstory"])

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
            data["role"] = self.role.value
        data["personality_traits"] = [trait.to_dict() for trait in self.personality_traits]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NPC":
        """Create NPC from dictionary."""
        if isinstance(data.get("role"), str):
            try:
                data["role"] = NPCRole(data["role"])
            except ValueError:
                data["role"] = NPCRole.CUSTOM
                data["custom_role"] = data.get("role")

        if "personality_traits" in data:
            traits = []
            for trait_data in data["personality_traits"]:
                if isinstance(trait_data, dict):
                    traits.append(PersonalityTrait(**trait_data))
            data["personality_traits"] = traits

        # Handle Character base class fields
        character_data = super().from_dict(data)
        return cls(**character_data.__dict__)

    def get_role_name(self) -> str:
        """Get the NPC's role name."""
        if self.role == NPCRole.CUSTOM:
            return self.custom_role or "Unknown"
        return self.role.value if self.role else "Unknown"


# Genre-specific specialized models

@dataclass
class CyberpunkAugmentation:
    """Cyberpunk-specific augmentation data."""
    
    name: str
    type: str  # "neural", "cyberlimb", "bioware", "nanotech"
    quality: str  # "street", "military", "corporate", "experimental"
    humanity_cost: int = 0
    description: str = ""
    abilities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SciFiTechnology:
    """Sci-fi specific technology/equipment."""
    
    name: str
    tech_level: int  # 1-10 scale
    category: str  # "weapon", "armor", "tool", "vehicle", "implant"
    power_source: str  # "fusion", "antimatter", "zero-point", "battery"
    description: str = ""
    capabilities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CosmicHorrorSanity:
    """Cosmic Horror sanity tracking."""
    
    current_sanity: int = 100
    max_sanity: int = 100
    indefinite_insanity: bool = False
    phobias: List[str] = field(default_factory=list)
    manias: List[str] = field(default_factory=list)
    encounters: List[str] = field(default_factory=list)  # Mythos encounters
    forbidden_knowledge: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PostApocalypticMutation:
    """Post-apocalyptic mutation data."""
    
    name: str
    type: str  # "physical", "mental", "metabolic", "sensory"
    severity: str  # "minor", "major", "extreme"
    description: str = ""
    benefits: List[str] = field(default_factory=list)
    drawbacks: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SuperheroPower:
    """Superhero power definition."""
    
    name: str
    category: str  # "physical", "energy", "mental", "magical", "tech"
    power_level: int  # 1-10 scale
    origin: str  # "mutation", "accident", "training", "alien", "magical", "tech"
    description: str = ""
    abilities: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class GenreSpecificData:
    """Container for genre-specific character data."""
    
    # Cyberpunk
    augmentations: List[CyberpunkAugmentation] = field(default_factory=list)
    street_cred: int = 0
    
    # Sci-Fi
    technologies: List[SciFiTechnology] = field(default_factory=list)
    ship_assignment: Optional[str] = None
    clearance_level: Optional[str] = None
    
    # Cosmic Horror
    sanity: Optional[CosmicHorrorSanity] = None
    cult_affiliations: List[str] = field(default_factory=list)
    
    # Post-Apocalyptic
    mutations: List[PostApocalypticMutation] = field(default_factory=list)
    radiation_resistance: int = 0
    survival_skills: List[str] = field(default_factory=list)
    
    # Superhero
    powers: List[SuperheroPower] = field(default_factory=list)
    secret_identity: Optional[str] = None
    nemesis: Optional[str] = None
    
    # Western
    reputation: str = "Unknown"  # "Unknown", "Greenhorn", "Known", "Famous", "Legendary"
    bounty: int = 0
    quick_draw: int = 0
    
    # Additional genre-specific data as dict
    custom_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "augmentations": [aug.to_dict() for aug in self.augmentations],
            "street_cred": self.street_cred,
            "technologies": [tech.to_dict() for tech in self.technologies],
            "ship_assignment": self.ship_assignment,
            "clearance_level": self.clearance_level,
            "sanity": self.sanity.to_dict() if self.sanity else None,
            "cult_affiliations": self.cult_affiliations,
            "mutations": [mut.to_dict() for mut in self.mutations],
            "radiation_resistance": self.radiation_resistance,
            "survival_skills": self.survival_skills,
            "powers": [power.to_dict() for power in self.powers],
            "secret_identity": self.secret_identity,
            "nemesis": self.nemesis,
            "reputation": self.reputation,
            "bounty": self.bounty,
            "quick_draw": self.quick_draw,
            "custom_data": self.custom_data,
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenreSpecificData":
        """Create from dictionary."""
        instance = cls()
        
        # Cyberpunk
        if "augmentations" in data:
            instance.augmentations = [
                CyberpunkAugmentation(**aug) for aug in data["augmentations"]
            ]
        instance.street_cred = data.get("street_cred", 0)
        
        # Sci-Fi
        if "technologies" in data:
            instance.technologies = [
                SciFiTechnology(**tech) for tech in data["technologies"]
            ]
        instance.ship_assignment = data.get("ship_assignment")
        instance.clearance_level = data.get("clearance_level")
        
        # Cosmic Horror
        if data.get("sanity"):
            instance.sanity = CosmicHorrorSanity(**data["sanity"])
        instance.cult_affiliations = data.get("cult_affiliations", [])
        
        # Post-Apocalyptic
        if "mutations" in data:
            instance.mutations = [
                PostApocalypticMutation(**mut) for mut in data["mutations"]
            ]
        instance.radiation_resistance = data.get("radiation_resistance", 0)
        instance.survival_skills = data.get("survival_skills", [])
        
        # Superhero
        if "powers" in data:
            instance.powers = [
                SuperheroPower(**power) for power in data["powers"]
            ]
        instance.secret_identity = data.get("secret_identity")
        instance.nemesis = data.get("nemesis")
        
        # Western
        instance.reputation = data.get("reputation", "Unknown")
        instance.bounty = data.get("bounty", 0)
        instance.quick_draw = data.get("quick_draw", 0)
        
        instance.custom_data = data.get("custom_data", {})
        
        return instance


@dataclass
class ExtendedCharacter(Character):
    """Extended character model with genre-specific data."""
    
    genre_data: GenreSpecificData = field(default_factory=GenreSpecificData)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including genre data."""
        data = super().to_dict()
        data["genre_data"] = self.genre_data.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtendedCharacter":
        """Create from dictionary including genre data."""
        # Handle genre data separately
        genre_data_dict = data.pop("genre_data", {})
        
        # Create base character
        base_char = super().from_dict(data)
        
        # Create extended character with all base fields
        extended = cls(**base_char.__dict__)
        
        # Add genre data
        if genre_data_dict:
            extended.genre_data = GenreSpecificData.from_dict(genre_data_dict)
        
        return extended
