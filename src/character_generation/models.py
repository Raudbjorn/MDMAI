"""Data models for character and NPC generation with enriched content."""

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


class CharacterTrait(Enum):
    """Comprehensive character traits from extracted content."""
    
    # Physical Traits
    AGILE = "agile"
    ATHLETIC = "athletic"
    BRAWNY = "brawny"
    BURLY = "burly"
    DELICATE = "delicate"
    DEXTEROUS = "dexterous"
    ENDURING = "enduring"
    ENERGETIC = "energetic"
    GRACEFUL = "graceful"
    HARDY = "hardy"
    LITHE = "lithe"
    MUSCULAR = "muscular"
    NIMBLE = "nimble"
    POWERFUL = "powerful"
    QUICK = "quick"
    RESILIENT = "resilient"
    ROBUST = "robust"
    RUGGED = "rugged"
    SCARRED = "scarred"
    SLENDER = "slender"
    STOCKY = "stocky"
    STRONG = "strong"
    STURDY = "sturdy"
    SWIFT = "swift"
    TALL = "tall"
    TOUGH = "tough"
    TOWERING = "towering"
    WEATHERED = "weathered"
    WIRY = "wiry"
    
    # Mental Traits
    ANALYTICAL = "analytical"
    ASTUTE = "astute"
    BRILLIANT = "brilliant"
    CALCULATING = "calculating"
    CLEVER = "clever"
    CREATIVE = "creative"
    CUNNING = "cunning"
    CURIOUS = "curious"
    FOCUSED = "focused"
    GENIUS = "genius"
    IMAGINATIVE = "imaginative"
    INSIGHTFUL = "insightful"
    INTELLECTUAL = "intellectual"
    INTELLIGENT = "intelligent"
    INTUITIVE = "intuitive"
    KNOWLEDGEABLE = "knowledgeable"
    LEARNED = "learned"
    LOGICAL = "logical"
    METHODICAL = "methodical"
    OBSERVANT = "observant"
    PERCEPTIVE = "perceptive"
    PHILOSOPHICAL = "philosophical"
    QUICK_WITTED = "quick_witted"
    RATIONAL = "rational"
    RESOURCEFUL = "resourceful"
    SCHOLARLY = "scholarly"
    SHARP = "sharp"
    SHREWD = "shrewd"
    STRATEGIC = "strategic"
    STUDIOUS = "studious"
    TACTICAL = "tactical"
    THOUGHTFUL = "thoughtful"
    WISE = "wise"
    
    # Emotional Traits
    AMBITIOUS = "ambitious"
    ANXIOUS = "anxious"
    BOLD = "bold"
    BRAVE = "brave"
    CALM = "calm"
    CAUTIOUS = "cautious"
    CHEERFUL = "cheerful"
    COMPASSIONATE = "compassionate"
    CONFIDENT = "confident"
    COURAGEOUS = "courageous"
    DETERMINED = "determined"
    DEVOTED = "devoted"
    DISCIPLINED = "disciplined"
    EMPATHETIC = "empathetic"
    ENTHUSIASTIC = "enthusiastic"
    FEARLESS = "fearless"
    FIERCE = "fierce"
    GENTLE = "gentle"
    GRIM = "grim"
    HOPEFUL = "hopeful"
    HUMBLE = "humble"
    IMPULSIVE = "impulsive"
    INDEPENDENT = "independent"
    JOYFUL = "joyful"
    KIND = "kind"
    LOYAL = "loyal"
    MELANCHOLIC = "melancholic"
    MERCIFUL = "merciful"
    PASSIONATE = "passionate"
    PATIENT = "patient"
    PROUD = "proud"
    REBELLIOUS = "rebellious"
    RECKLESS = "reckless"
    RESOLUTE = "resolute"
    RUTHLESS = "ruthless"
    SELFLESS = "selfless"
    SERENE = "serene"
    SINCERE = "sincere"
    SKEPTICAL = "skeptical"
    STEADFAST = "steadfast"
    STOIC = "stoic"
    STUBBORN = "stubborn"
    SYMPATHETIC = "sympathetic"
    TENACIOUS = "tenacious"
    VENGEFUL = "vengeful"
    VIGILANT = "vigilant"
    VOLATILE = "volatile"
    ZEALOUS = "zealous"
    
    # Social Traits
    CHARISMATIC = "charismatic"
    CHARMING = "charming"
    DIPLOMATIC = "diplomatic"
    ELOQUENT = "eloquent"
    GREGARIOUS = "gregarious"
    INTIMIDATING = "intimidating"
    MYSTERIOUS = "mysterious"
    PERSUASIVE = "persuasive"
    RESERVED = "reserved"
    SHY = "shy"
    SOCIABLE = "sociable"
    WITTY = "witty"


class CharacterBackground(Enum):
    """Expanded character backgrounds from extracted content."""
    
    # Traditional Backgrounds
    ACOLYTE = "acolyte"
    CRIMINAL = "criminal"
    FOLK_HERO = "folk_hero"
    NOBLE = "noble"
    SAGE = "sage"
    SOLDIER = "soldier"
    HERMIT = "hermit"
    ENTERTAINER = "entertainer"
    GUILD_ARTISAN = "guild_artisan"
    OUTLANDER = "outlander"
    SAILOR = "sailor"
    URCHIN = "urchin"
    
    # Expanded Traditional
    ALCHEMIST = "alchemist"
    AMBASSADOR = "ambassador"
    ARISTOCRAT = "aristocrat"
    ASSASSIN = "assassin"
    BANDIT = "bandit"
    BLACKSMITH = "blacksmith"
    BOUNTY_HUNTER = "bounty_hunter"
    CARAVAN_GUARD = "caravan_guard"
    CHARLATAN = "charlatan"
    CLAN_CRAFTER = "clan_crafter"
    CLOISTERED_SCHOLAR = "cloistered_scholar"
    COURTIER = "courtier"
    CULT_INITIATE = "cult_initiate"
    DETECTIVE = "detective"
    EXILE = "exile"
    EXPLORER = "explorer"
    FARMER = "farmer"
    FISHER = "fisher"
    GLADIATOR = "gladiator"
    GUARD = "guard"
    HEALER = "healer"
    HUNTER = "hunter"
    INNKEEPER = "innkeeper"
    INVESTIGATOR = "investigator"
    KNIGHT = "knight"
    LIBRARIAN = "librarian"
    MERCHANT = "merchant"
    MERCENARY = "mercenary"
    MINER = "miner"
    MONK_INITIATE = "monk_initiate"
    MYSTIC = "mystic"
    PIRATE = "pirate"
    PRIEST = "priest"
    RANGER_SCOUT = "ranger_scout"
    REFUGEE = "refugee"
    SCHOLAR_MAGE = "scholar_mage"
    SCRIBE = "scribe"
    SHIP_CAPTAIN = "ship_captain"
    SMUGGLER = "smuggler"
    SPY = "spy"
    STREET_THIEF = "street_thief"
    TAVERN_KEEPER = "tavern_keeper"
    TEMPLE_GUARDIAN = "temple_guardian"
    THIEF = "thief"
    TRADER = "trader"
    TRIBAL_WARRIOR = "tribal_warrior"
    VETERAN = "veteran"
    WANDERER = "wanderer"
    WAR_REFUGEE = "war_refugee"
    
    # Sci-Fi Backgrounds
    ASTEROID_MINER = "asteroid_miner"
    COLONY_ADMINISTRATOR = "colony_administrator"
    CORPORATE_AGENT = "corporate_agent"
    CYBORG_ENGINEER = "cyborg_engineer"
    DATA_ANALYST = "data_analyst"
    DIPLOMAT_ENVOY = "diplomat_envoy"
    GENETIC_RESEARCHER = "genetic_researcher"
    HACKER = "hacker"
    JUMP_PILOT = "jump_pilot"
    ORBITAL_MECHANIC = "orbital_mechanic"
    SPACE_MARINE = "space_marine"
    STARSHIP_ENGINEER = "starship_engineer"
    TERRAFORMER = "terraformer"
    VOID_TRADER = "void_trader"
    XENOBIOLOGIST = "xenobiologist"
    
    # Cyberpunk Backgrounds
    CORPORATE_EXEC = "corporate_exec"
    FIXER = "fixer"
    GANG_MEMBER = "gang_member"
    MEDIA_JOURNALIST = "media_journalist"
    NETRUNNER = "netrunner"
    NOMAD = "nomad"
    RIPPERDOC = "ripperdoc"
    ROCKERBOY = "rockerboy"
    STREET_SAMURAI = "street_samurai"
    TECH_SPECIALIST = "tech_specialist"
    
    # Post-Apocalyptic Backgrounds
    BUNKER_SURVIVOR = "bunker_survivor"
    CARAVAN_TRADER = "caravan_trader"
    MUTANT_OUTCAST = "mutant_outcast"
    RAIDER = "raider"
    SCAVENGER = "scavenger"
    SETTLEMENT_LEADER = "settlement_leader"
    TRIBAL_SHAMAN = "tribal_shaman"
    VAULT_DWELLER = "vault_dweller"
    WASTELAND_DOCTOR = "wasteland_doctor"
    WASTELAND_SCOUT = "wasteland_scout"
    
    # Cosmic Horror Backgrounds
    ANTIQUARIAN = "antiquarian"
    ASYLUM_PATIENT = "asylum_patient"
    CULT_SURVIVOR = "cult_survivor"
    CURSED_BLOODLINE = "cursed_bloodline"
    DREAM_TOUCHED = "dream_touched"
    OCCULT_INVESTIGATOR = "occult_investigator"
    PROFESSOR = "professor"
    PSYCHIC_SENSITIVE = "psychic_sensitive"
    
    # Western Backgrounds
    BOUNTY_KILLER = "bounty_killer"
    CATTLE_RUSTLER = "cattle_rustler"
    FRONTIER_DOCTOR = "frontier_doctor"
    GUNSLINGER = "gunslinger"
    HOMESTEADER = "homesteader"
    LAWMAN = "lawman"
    OUTLAW = "outlaw"
    PREACHER = "preacher"
    PROSPECTOR = "prospector"
    RANCH_HAND = "ranch_hand"
    SALOON_OWNER = "saloon_owner"
    STAGE_DRIVER = "stage_driver"
    
    # Superhero Backgrounds
    ALIEN_REFUGEE = "alien_refugee"
    GOVERNMENT_AGENT = "government_agent"
    LAB_ACCIDENT_SURVIVOR = "lab_accident_survivor"
    MASKED_VIGILANTE = "masked_vigilante"
    MILITARY_EXPERIMENT = "military_experiment"
    MUTANT_ACTIVIST = "mutant_activist"
    REPORTER = "reporter"
    SCIENTIST = "scientist"
    SIDEKICK = "sidekick"
    TECH_GENIUS = "tech_genius"


class CharacterMotivation(Enum):
    """Character motivations from extracted content."""
    
    # Core Desires
    ACCEPTANCE = "acceptance"
    ACHIEVEMENT = "achievement"
    ADVENTURE = "adventure"
    APPROVAL = "approval"
    BALANCE = "balance"
    BELONGING = "belonging"
    CHALLENGE = "challenge"
    CHANGE = "change"
    COMFORT = "comfort"
    CONNECTION = "connection"
    CONTROL = "control"
    DISCOVERY = "discovery"
    DUTY = "duty"
    EXCELLENCE = "excellence"
    EXCITEMENT = "excitement"
    EXPLORATION = "exploration"
    FAME = "fame"
    FREEDOM = "freedom"
    GLORY = "glory"
    GROWTH = "growth"
    HAPPINESS = "happiness"
    HARMONY = "harmony"
    HONOR = "honor"
    HOPE = "hope"
    INDEPENDENCE = "independence"
    INFLUENCE = "influence"
    JUSTICE = "justice"
    KNOWLEDGE = "knowledge"
    LEGACY = "legacy"
    LOVE = "love"
    MASTERY = "mastery"
    MEANING = "meaning"
    ORDER = "order"
    PEACE = "peace"
    PERFECTION = "perfection"
    PLEASURE = "pleasure"
    POWER = "power"
    PRESTIGE = "prestige"
    PROGRESS = "progress"
    PROSPERITY = "prosperity"
    PROTECTION = "protection"
    PURPOSE = "purpose"
    RECOGNITION = "recognition"
    REDEMPTION = "redemption"
    RESPECT = "respect"
    RESTORATION = "restoration"
    REVENGE = "revenge"
    SAFETY = "safety"
    SALVATION = "salvation"
    SECURITY = "security"
    SERVICE = "service"
    STABILITY = "stability"
    STATUS = "status"
    STRENGTH = "strength"
    SUCCESS = "success"
    SURVIVAL = "survival"
    TRADITION = "tradition"
    TRANSCENDENCE = "transcendence"
    TRANSFORMATION = "transformation"
    TRUTH = "truth"
    UNDERSTANDING = "understanding"
    UNITY = "unity"
    VALIDATION = "validation"
    VENGEANCE = "vengeance"
    VICTORY = "victory"
    WEALTH = "wealth"
    WISDOM = "wisdom"
    
    # Fears
    ABANDONMENT = "abandonment"
    BETRAYAL = "betrayal"
    CHAOS = "chaos"
    CONFINEMENT = "confinement"
    CORRUPTION = "corruption"
    DARKNESS = "darkness"
    DEATH = "death"
    DEFEAT = "defeat"
    DISGRACE = "disgrace"
    DISEASE = "disease"
    EXPOSURE = "exposure"
    FAILURE = "failure"
    FORGETTING = "forgetting"
    HELPLESSNESS = "helplessness"
    HUMILIATION = "humiliation"
    IGNORANCE = "ignorance"
    INSIGNIFICANCE = "insignificance"
    ISOLATION = "isolation"
    LOSS = "loss"
    MADNESS = "madness"
    MEANINGLESSNESS = "meaninglessness"
    OBSCURITY = "obscurity"
    PAIN = "pain"
    POVERTY = "poverty"
    POWERLESSNESS = "powerlessness"
    REJECTION = "rejection"
    RESPONSIBILITY = "responsibility"
    STAGNATION = "stagnation"
    SUFFERING = "suffering"
    THE_UNKNOWN = "the_unknown"
    VULNERABILITY = "vulnerability"
    WEAKNESS = "weakness"
    
    # Complex Motivations
    ATONEMENT = "atonement"
    BREAKING_CHAINS = "breaking_chains"
    BUILDING_EMPIRE = "building_empire"
    CHASING_LEGEND = "chasing_legend"
    CLAIMING_BIRTHRIGHT = "claiming_birthright"
    CONQUERING_FEAR = "conquering_fear"
    DEFENDING_HOMELAND = "defending_homeland"
    DESTROYING_EVIL = "destroying_evil"
    DISCOVERING_HERITAGE = "discovering_heritage"
    ENDING_TYRANNY = "ending_tyranny"
    ESCAPING_PAST = "escaping_past"
    FINDING_HOME = "finding_home"
    FINDING_IDENTITY = "finding_identity"
    FULFILLING_DESTINY = "fulfilling_destiny"
    FULFILLING_OATH = "fulfilling_oath"
    HONORING_ANCESTORS = "honoring_ancestors"
    LIBERATING_OPPRESSED = "liberating_oppressed"
    MAINTAINING_BALANCE = "maintaining_balance"
    MAKING_AMENDS = "making_amends"
    PRESERVING_MEMORY = "preserving_memory"
    PROTECTING_INNOCENT = "protecting_innocent"
    PROVING_WORTH = "proving_worth"
    RECLAIMING_THRONE = "reclaiming_throne"
    RECOVERING_ARTIFACT = "recovering_artifact"
    REDEEMING_FAMILY = "redeeming_family"
    RESTORING_HONOR = "restoring_honor"
    REUNITING_FAMILY = "reuniting_family"
    REVEALING_TRUTH = "revealing_truth"
    SAVING_LOVED_ONE = "saving_loved_one"
    SEEKING_ENLIGHTENMENT = "seeking_enlightenment"
    SOLVING_MYSTERY = "solving_mystery"
    STOPPING_PROPHECY = "stopping_prophecy"
    UNCOVERING_CONSPIRACY = "uncovering_conspiracy"
    UNITING_PEOPLE = "uniting_people"


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


class WeaponType(Enum):
    """Expanded weapon types from extracted content."""
    
    # Melee Weapons
    SWORD = "sword"
    LONGSWORD = "longsword"
    SHORTSWORD = "shortsword"
    GREATSWORD = "greatsword"
    RAPIER = "rapier"
    SCIMITAR = "scimitar"
    KATANA = "katana"
    CUTLASS = "cutlass"
    BROADSWORD = "broadsword"
    CLAYMORE = "claymore"
    
    AXE = "axe"
    BATTLEAXE = "battleaxe"
    HANDAXE = "handaxe"
    GREATAXE = "greataxe"
    WARAXE = "waraxe"
    THROWING_AXE = "throwing_axe"
    
    MACE = "mace"
    CLUB = "club"
    WARHAMMER = "warhammer"
    MAUL = "maul"
    MORNINGSTAR = "morningstar"
    FLAIL = "flail"
    
    SPEAR = "spear"
    PIKE = "pike"
    HALBERD = "halberd"
    GLAIVE = "glaive"
    TRIDENT = "trident"
    LANCE = "lance"
    
    DAGGER = "dagger"
    KNIFE = "knife"
    STILETTO = "stiletto"
    DIRK = "dirk"
    KRIS = "kris"
    
    STAFF = "staff"
    QUARTERSTAFF = "quarterstaff"
    WALKING_STICK = "walking_stick"
    BO_STAFF = "bo_staff"
    
    # Ranged Weapons
    BOW = "bow"
    LONGBOW = "longbow"
    SHORTBOW = "shortbow"
    COMPOSITE_BOW = "composite_bow"
    CROSSBOW = "crossbow"
    HEAVY_CROSSBOW = "heavy_crossbow"
    HAND_CROSSBOW = "hand_crossbow"
    
    # Firearms
    PISTOL = "pistol"
    REVOLVER = "revolver"
    RIFLE = "rifle"
    SHOTGUN = "shotgun"
    MUSKET = "musket"
    BLUNDERBUSS = "blunderbuss"
    
    # Energy Weapons
    LASER_PISTOL = "laser_pistol"
    LASER_RIFLE = "laser_rifle"
    PLASMA_GUN = "plasma_gun"
    PULSE_RIFLE = "pulse_rifle"
    DISRUPTOR = "disruptor"
    STUNNER = "stunner"
    
    # Exotic Weapons
    WHIP = "whip"
    NET = "net"
    BOLAS = "bolas"
    CHAKRAM = "chakram"
    SHURIKEN = "shuriken"
    BLOWGUN = "blowgun"
    
    # Improvised/Natural
    CLAWS = "claws"
    BITE = "bite"
    TENTACLE = "tentacle"
    IMPROVISED = "improvised"


class ItemType(Enum):
    """Expanded item types from extracted content."""
    
    # Adventuring Gear
    ROPE = "rope"
    GRAPPLING_HOOK = "grappling_hook"
    TORCH = "torch"
    LANTERN = "lantern"
    OIL_FLASK = "oil_flask"
    TINDERBOX = "tinderbox"
    BEDROLL = "bedroll"
    TENT = "tent"
    RATIONS = "rations"
    WATERSKIN = "waterskin"
    BACKPACK = "backpack"
    POUCH = "pouch"
    SACK = "sack"
    CHEST = "chest"
    BARREL = "barrel"
    
    # Tools
    THIEVES_TOOLS = "thieves_tools"
    LOCKPICKS = "lockpicks"
    CROWBAR = "crowbar"
    HAMMER = "hammer"
    PITON = "piton"
    SHOVEL = "shovel"
    PICKAXE = "pickaxe"
    CLIMBING_KIT = "climbing_kit"
    HEALERS_KIT = "healers_kit"
    HERBALISM_KIT = "herbalism_kit"
    ALCHEMIST_SUPPLIES = "alchemist_supplies"
    BREWERS_SUPPLIES = "brewers_supplies"
    CALLIGRAPHERS_SUPPLIES = "calligraphers_supplies"
    CARPENTERS_TOOLS = "carpenters_tools"
    CARTOGRAPHERS_TOOLS = "cartographers_tools"
    COBBLERS_TOOLS = "cobblers_tools"
    COOKS_UTENSILS = "cooks_utensils"
    GLASSBLOWERS_TOOLS = "glassblowers_tools"
    JEWELERS_TOOLS = "jewelers_tools"
    LEATHERWORKERS_TOOLS = "leatherworkers_tools"
    MASONS_TOOLS = "masons_tools"
    PAINTERS_SUPPLIES = "painters_supplies"
    POTTERS_TOOLS = "potters_tools"
    SMITHS_TOOLS = "smiths_tools"
    TINKERS_TOOLS = "tinkers_tools"
    WEAVERS_TOOLS = "weavers_tools"
    WOODCARVERS_TOOLS = "woodcarvers_tools"
    DISGUISE_KIT = "disguise_kit"
    FORGERY_KIT = "forgery_kit"
    GAMING_SET = "gaming_set"
    MUSICAL_INSTRUMENT = "musical_instrument"
    NAVIGATORS_TOOLS = "navigators_tools"
    POISONERS_KIT = "poisoners_kit"
    
    # Magic Items
    POTION = "potion"
    SCROLL = "scroll"
    WAND = "wand"
    ROD = "rod"
    STAFF_MAGICAL = "staff_magical"
    RING = "ring"
    AMULET = "amulet"
    CLOAK = "cloak"
    BOOTS = "boots"
    GLOVES = "gloves"
    BELT = "belt"
    BRACERS = "bracers"
    CIRCLET = "circlet"
    
    # Books and Documents
    SPELLBOOK = "spellbook"
    TOME = "tome"
    MAP = "map"
    LETTER = "letter"
    JOURNAL = "journal"
    CONTRACT = "contract"
    DEED = "deed"
    
    # Technology Items
    COMMUNICATOR = "communicator"
    SCANNER = "scanner"
    DATAPAD = "datapad"
    HOLO_PROJECTOR = "holo_projector"
    MEDKIT = "medkit"
    REPAIR_KIT = "repair_kit"
    ENERGY_CELL = "energy_cell"
    CYBERDECK = "cyberdeck"
    NEURAL_IMPLANT = "neural_implant"
    
    # Miscellaneous
    HOLY_SYMBOL = "holy_symbol"
    COMPONENT_POUCH = "component_pouch"
    ARCANE_FOCUS = "arcane_focus"
    DRUIDIC_FOCUS = "druidic_focus"
    MIRROR = "mirror"
    MAGNIFYING_GLASS = "magnifying_glass"
    SPYGLASS = "spyglass"
    COMPASS = "compass"
    HOURGLASS = "hourglass"
    SCALES = "scales"
    VIAL = "vial"
    FLASK = "flask"
    BOTTLE = "bottle"
    SOAP = "soap"
    BELL = "bell"
    WHISTLE = "whistle"
    SIGNAL_HORN = "signal_horn"
    MANACLES = "manacles"
    CHAIN = "chain"
    BALL_BEARINGS = "ball_bearings"
    CALTROPS = "caltrops"
    CHALK = "chalk"
    LADDER = "ladder"
    POLE = "pole"
    RAM = "ram"
    SIGNET_RING = "signet_ring"
    SEALING_WAX = "sealing_wax"


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
class StoryHook:
    """Story hook for character integration into narratives."""
    
    hook_type: str  # "quest", "mystery", "conflict", "discovery", "relationship"
    title: str
    description: str
    urgency: str = "medium"  # "low", "medium", "high", "critical"
    stakes: str = ""
    potential_allies: List[str] = field(default_factory=list)
    potential_enemies: List[str] = field(default_factory=list)
    rewards: List[str] = field(default_factory=list)
    complications: List[str] = field(default_factory=list)
    genre_tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert story hook to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoryHook":
        """Create story hook from dictionary."""
        return cls(**data)


@dataclass
class WorldElement:
    """World-building element for character context."""
    
    element_type: str  # "location", "faction", "event", "artifact", "phenomenon"
    name: str
    description: str
    significance: str = ""
    history: str = ""
    current_state: str = ""
    connections: List[str] = field(default_factory=list)
    secrets: List[str] = field(default_factory=list)
    rumors: List[str] = field(default_factory=list)
    genre: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert world element to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorldElement":
        """Create world element from dictionary."""
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
    
    # Story integration
    story_hooks: List[StoryHook] = field(default_factory=list)
    world_connections: List[WorldElement] = field(default_factory=list)

    # System/personality-aware elements
    narrative_style: str = ""  # Matches source personality
    cultural_references: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert backstory to dictionary."""
        data = asdict(self)
        data["story_hooks"] = [hook.to_dict() for hook in self.story_hooks]
        data["world_connections"] = [elem.to_dict() for elem in self.world_connections]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Backstory":
        """Create backstory from dictionary."""
        if "story_hooks" in data:
            data["story_hooks"] = [StoryHook.from_dict(h) for h in data["story_hooks"]]
        if "world_connections" in data:
            data["world_connections"] = [WorldElement.from_dict(e) for e in data["world_connections"]]
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
    genre: Optional[TTRPGGenre] = None  # Genre category

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
        # Extract genre_data before passing to parent
        data_copy = data.copy()
        genre_data_dict = data_copy.pop("genre_data", {})

        # Call parent from_dict to get base character data
        # Since we're using cls, it will create an ExtendedCharacter instance
        instance = super().from_dict(data_copy)

        if genre_data_dict:
            instance.genre_data = GenreSpecificData.from_dict(genre_data_dict)

        return instance
