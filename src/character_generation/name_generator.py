"""Genre-specific name generation system for TTRPG characters and NPCs."""

import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .models import CharacterRace, NPCRole, TTRPGGenre

logger = logging.getLogger(__name__)


class NameStyle(Enum):
    """Different naming styles for various contexts."""
    
    FORMAL = "formal"
    CASUAL = "casual"
    NICKNAME = "nickname"
    TITLE = "title"
    ALIAS = "alias"
    CODENAME = "codename"
    STREET = "street"
    CORPORATE = "corporate"
    MILITARY = "military"
    MYSTICAL = "mystical"


@dataclass
class NameComponents:
    """Components that make up a generated name."""
    
    first_name: str
    last_name: Optional[str] = None
    title: Optional[str] = None
    nickname: Optional[str] = None
    suffix: Optional[str] = None
    
    def get_full_name(self, style: NameStyle = NameStyle.FORMAL) -> str:
        """Get the full name formatted according to style."""
        if style == NameStyle.FORMAL:
            parts = []
            if self.title:
                parts.append(self.title)
            parts.append(self.first_name)
            if self.last_name:
                parts.append(self.last_name)
            if self.suffix:
                parts.append(self.suffix)
            return " ".join(parts)
        elif style == NameStyle.CASUAL:
            return self.first_name
        elif style == NameStyle.NICKNAME and self.nickname:
            return self.nickname
        elif style == NameStyle.ALIAS:
            if self.nickname:
                return f'"{self.nickname}" {self.last_name or self.first_name}'
            return self.get_full_name(NameStyle.FORMAL)
        else:
            return self.get_full_name(NameStyle.FORMAL)


class NameGenerator:
    """Generate genre and context-appropriate names for TTRPG characters."""
    
    # Fantasy name pools
    FANTASY_NAMES = {
        "first_male": ["Aldric", "Theron", "Marcus", "Gareth", "Darius", "Lysander", 
                       "Caspian", "Rowan", "Felix", "Lucian", "Dorian", "Silas"],
        "first_female": ["Elena", "Lyra", "Mira", "Seraphina", "Aurora", "Celeste",
                        "Iris", "Luna", "Nova", "Ophelia", "Thalia", "Violet"],
        "first_neutral": ["Morgan", "Sage", "River", "Phoenix", "Ash", "Storm",
                         "Raven", "Sky", "Winter", "Vale", "Quinn", "Echo"],
        "last_common": ["Blackwood", "Stormwind", "Ironforge", "Goldshire", "Silverleaf",
                       "Brightblade", "Shadowbane", "Moonwhisper", "Starweaver", "Flameheart"],
        "last_noble": ["von Ravencrest", "de Montrose", "Blackstone", "Winterhold",
                      "Dragonmoor", "Ashford", "Thornwood", "Greycastle"],
        "titles": ["Sir", "Lady", "Lord", "Dame", "Master", "Mistress", "Baron", "Baroness"],
        "nicknames": ["the Bold", "the Wise", "Shadowstep", "Brightblade", "Ironhand",
                     "the Swift", "Stormcaller", "the Cunning"]
    }
    
    # Sci-Fi name pools
    SCIFI_NAMES = {
        "first_male": ["Nova", "Orion", "Atlas", "Zephyr", "Kai", "Leo", "Phoenix",
                       "Axel", "Cyrus", "Rex", "Jax", "Zane"],
        "first_female": ["Luna", "Vega", "Stella", "Astra", "Lyra", "Cora", "Zara",
                        "Nyx", "Echo", "Aria", "Sage", "Iris"],
        "first_neutral": ["Sky", "River", "Storm", "Ash", "Onyx", "Raven", "Frost",
                         "Blaze", "Shadow", "Spark", "Vale", "Zenith"],
        "last_common": ["Stardust", "Cosmos", "Nebula", "Quasar", "Pulsar", "Void",
                       "Stellar", "Photon", "Quantum", "Nexus", "Vector", "Prime"],
        "last_colony": ["Mars-7", "Terra-Prime", "Alpha-3", "Sigma-9", "Omega-X",
                       "Delta-4", "Epsilon-2", "Gamma-5", "Beta-8"],
        "titles": ["Captain", "Commander", "Doctor", "Professor", "Admiral", "Lieutenant",
                  "Specialist", "Engineer", "Pilot", "Navigator"],
        "designations": ["X-", "Z-", "Q-", "V-", "K-", "N-", "R-", "T-"]
    }
    
    # Cyberpunk name pools
    CYBERPUNK_NAMES = {
        "first_male": ["Neon", "Chrome", "Razor", "Ghost", "Binary", "Cipher",
                       "Dex", "Volt", "Glitch", "Static", "Vector", "Pixel"],
        "first_female": ["Shadow", "Hex", "Nova", "Synth", "Echo", "Viper",
                        "Flux", "Nyx", "Spark", "Crash", "Wire", "Phoenix"],
        "first_neutral": ["Zero", "Byte", "Code", "Virus", "Daemon", "Script",
                         "Cache", "Proxy", "Socket", "Token", "Hash", "Root"],
        "last_street": ["Runner", "Jack", "Burn", "Wire", "Crash", "Hex", "Byte",
                       "Ghost", "Shadow", "Chrome", "Neon", "Static"],
        "last_corporate": ["Arasaka", "Militech", "Zetatech", "Biotechnica", "Petrochem",
                          "Orbital", "Dynacorp", "Microtech", "Raven", "Lazarus"],
        "handles": ["CrashOverride", "AcidBurn", "ZeroCool", "ThePhantom", "NightCrawler",
                   "GhostInShell", "NeuralBurn", "DataThief", "NetDemon", "ByteBandit"],
        "titles": ["Netrunner", "Solo", "Fixer", "Techie", "Nomad", "Corpo", "Edgerunner"]
    }
    
    # Cosmic Horror name pools
    COSMIC_HORROR_NAMES = {
        "first_male": ["Randolph", "Herbert", "Wilbur", "Silas", "Ephraim", "Jeremiah",
                       "Barnabas", "Obadiah", "Ezekiel", "Thaddeus", "Ambrose", "Cornelius"],
        "first_female": ["Lavinia", "Prudence", "Constance", "Temperance", "Mercy", "Patience",
                        "Charity", "Verity", "Felicity", "Agatha", "Minerva", "Cordelia"],
        "first_neutral": ["Morgan", "Avery", "Quinn", "Harper", "Sage", "River",
                         "August", "Winter", "Salem", "Raven", "Gray", "Ash"],
        "last_new_england": ["Whateley", "Armitage", "Marsh", "Gilman", "Ward", "Carter",
                             "Pickman", "Wilmarth", "Akeley", "Derby", "Peaslee", "Olmstead"],
        "last_academic": ["Professor", "Doctor", "Reverend", "Dean", "Scholar"],
        "titles": ["Professor", "Doctor", "Reverend", "Inspector", "Detective", "Librarian"],
        "epithets": ["the Mad", "the Cursed", "the Touched", "the Dreamer", "the Lost",
                     "the Damned", "the Seeker", "the Wanderer"]
    }
    
    # Post-Apocalyptic name pools
    POST_APOCALYPTIC_NAMES = {
        "first_male": ["Ash", "Rust", "Storm", "Dust", "Hawk", "Wolf", "Stone",
                       "Blade", "Rex", "Max", "Tank", "Diesel"],
        "first_female": ["Raven", "Scar", "Nova", "Echo", "Fury", "Viper", "Phoenix",
                        "Storm", "Blaze", "Shadow", "Thorn", "Jade"],
        "first_neutral": ["Ghost", "Reaper", "Scrap", "Zero", "Rad", "Waste",
                         "Grit", "Spike", "Bolt", "Wire", "Chrome", "Nuke"],
        "last_descriptive": ["Walker", "Survivor", "Scav", "Hunter", "Wanderer", "Keeper",
                            "Runner", "Fighter", "Breaker", "Trader", "Guard", "Scout"],
        "last_origin": ["Vault-13", "Megaton", "Rivet", "Diamond", "Paradise", "Oasis",
                       "Citadel", "Wasteland", "Bunker", "Outpost", "Ruins", "Crater"],
        "nicknames": ["Two-Shot", "Dogmeat", "Grognak", "Psycho", "Jet", "Radroach",
                     "Deathclaw", "Smoothskin", "Ghoul", "Mutie", "Wastelander"],
        "titles": ["Elder", "Overseer", "Warlord", "Chief", "Boss", "Alpha"]
    }
    
    # Western name pools
    WESTERN_NAMES = {
        "first_male": ["Jesse", "Wyatt", "Doc", "Billy", "Frank", "Butch", "Cole",
                       "Jake", "Luke", "Wade", "Clay", "Hank"],
        "first_female": ["Belle", "Calamity", "Annie", "Rose", "Pearl", "Daisy",
                        "Grace", "Sally", "Kate", "Lilly", "Ruby", "May"],
        "first_neutral": ["Dakota", "Casey", "Morgan", "Jordan", "Taylor", "Jamie",
                         "Quinn", "Riley", "Sage", "River", "Sky", "August"],
        "last_common": ["Morgan", "Earp", "Holliday", "James", "Cassidy", "Starr",
                       "Black", "Stone", "Walker", "Rider", "Turner", "Miller"],
        "last_descriptive": ["the Kid", "the Preacher", "the Gambler", "the Gunslinger",
                            "the Marshal", "the Outlaw", "the Drifter", "the Stranger"],
        "nicknames": ["Wild", "Dead-Eye", "Quick-Draw", "Six-Gun", "Rattlesnake",
                     "Maverick", "Bronco", "Mustang", "Coyote", "Vulture"],
        "titles": ["Sheriff", "Marshal", "Deputy", "Judge", "Reverend", "Doc"]
    }
    
    # Superhero name pools
    SUPERHERO_NAMES = {
        "first_male": ["Max", "Victor", "Bruce", "Clark", "Peter", "Tony", "Steve",
                       "Wade", "Logan", "Scott", "Matt", "Barry"],
        "first_female": ["Diana", "Jean", "Natasha", "Wanda", "Carol", "Jessica",
                        "Barbara", "Selina", "Emma", "Ororo", "Kitty", "Sue"],
        "first_neutral": ["Alex", "Jordan", "Casey", "Morgan", "Quinn", "Riley",
                         "Sam", "Taylor", "Jamie", "Drew", "Robin", "Blake"],
        "last_common": ["Power", "Steel", "Storm", "Knight", "Swift", "Strong",
                       "Bright", "Dark", "Shadow", "Light", "Force", "Guard"],
        "hero_names": ["Captain", "Doctor", "Mister", "Miss", "The", "Agent"],
        "hero_suffixes": ["man", "woman", "girl", "boy", "lord", "lady"],
        "titles": ["Captain", "Doctor", "Professor", "Agent", "Commander", "General"]
    }
    
    # Race-specific name modifications
    RACE_NAME_MODIFIERS = {
        CharacterRace.ELF: {
            "prefixes": ["Sil", "Gal", "El", "Ar", "Leg", "Thaur"],
            "suffixes": ["wen", "riel", "iel", "dor", "las", "ion"],
            "patterns": ["flowing", "musical", "nature-themed"]
        },
        CharacterRace.DWARF: {
            "prefixes": ["Thor", "Gim", "Bal", "Dur", "Oin", "Dwal"],
            "suffixes": ["in", "li", "ori", "oin", "rim", "din"],
            "patterns": ["strong", "stone-themed", "forge-related"]
        },
        CharacterRace.HALFLING: {
            "prefixes": ["Pip", "Mer", "Sam", "Fro", "Bill", "Rose"],
            "suffixes": ["pins", "wise", "foot", "gins", "took", "bottom"],
            "patterns": ["comfortable", "food-related", "homely"]
        },
        CharacterRace.ORC: {
            "prefixes": ["Gro", "Ug", "Mog", "Grim", "Gor", "Kro"],
            "suffixes": ["bash", "tooth", "jaw", "skull", "bone", "blood"],
            "patterns": ["harsh", "guttural", "violent"]
        }
    }
    
    # NPC Role-specific name patterns
    NPC_ROLE_PATTERNS = {
        NPCRole.MERCHANT: {
            "titles": ["Master", "Mistress", "Goodman", "Goodwife"],
            "suffixes": ["the Trader", "of the Market", "Coinpurse", "Goldhand"],
            "style": NameStyle.FORMAL
        },
        NPCRole.GUARD: {
            "titles": ["Sergeant", "Captain", "Watchman", "Guard"],
            "suffixes": ["the Vigilant", "Ironhand", "the Watcher", "Shieldbearer"],
            "style": NameStyle.MILITARY
        },
        NPCRole.NOBLE: {
            "titles": ["Lord", "Lady", "Baron", "Baroness", "Count", "Countess"],
            "suffixes": ["of [Location]", "the [Ordinal]", "von [Family]"],
            "style": NameStyle.FORMAL
        },
        NPCRole.CRIMINAL: {
            "titles": ["", ""],  # Often no title
            "suffixes": ["the Knife", "Fingers", "the Shadow", "Quick"],
            "style": NameStyle.ALIAS
        },
        NPCRole.SCHOLAR: {
            "titles": ["Scholar", "Sage", "Master", "Doctor", "Professor"],
            "suffixes": ["the Learned", "the Wise", "of the Tower", "Scrollkeeper"],
            "style": NameStyle.FORMAL
        },
        NPCRole.PRIEST: {
            "titles": ["Father", "Mother", "Brother", "Sister", "Reverend"],
            "suffixes": ["the Blessed", "the Devout", "of the Light", "Faithful"],
            "style": NameStyle.FORMAL
        }
    }
    
    @classmethod
    def generate_name(
        cls,
        genre: TTRPGGenre = TTRPGGenre.FANTASY,
        gender: Optional[str] = None,
        race: Optional[CharacterRace] = None,
        role: Optional[NPCRole] = None,
        style: NameStyle = NameStyle.FORMAL,
        include_title: bool = False,
        include_nickname: bool = False
    ) -> Tuple[str, NameComponents]:
        """
        Generate a genre and context-appropriate name.
        
        Args:
            genre: The TTRPG genre for name generation
            gender: 'male', 'female', or 'neutral' (random if None)
            race: Character race for race-specific naming
            role: NPC role for role-specific naming
            style: The naming style to use
            include_title: Whether to include a title
            include_nickname: Whether to include a nickname
            
        Returns:
            Either a string name or tuple of (name_string, NameComponents)
        """
        # Determine gender if not specified
        if gender is None:
            gender = random.choice(["male", "female", "neutral"])
        
        # Get the appropriate name pool
        name_pool = cls._get_name_pool(genre)
        
        # Generate base components
        first_name = cls._generate_first_name(name_pool, gender, race)
        last_name = cls._generate_last_name(name_pool, genre, race, role)
        
        # Generate optional components
        title = None
        if include_title:
            title = cls._generate_title(name_pool, genre, role, gender)
        
        nickname = None
        if include_nickname:
            nickname = cls._generate_nickname(name_pool, genre, race, role)
        
        # Apply role-specific patterns
        if role:
            title, nickname = cls._apply_role_patterns(role, title, nickname, genre)
        
        # Create name components
        components = NameComponents(
            first_name=first_name,
            last_name=last_name,
            title=title,
            nickname=nickname
        )
        
        # Return formatted name
        return components.get_full_name(style), components
    
    @classmethod
    def generate_batch(
        cls,
        count: int,
        genre: TTRPGGenre = TTRPGGenre.FANTASY,
        **kwargs
    ) -> List[str]:
        """Generate multiple names with the same parameters."""
        names = []
        for _ in range(count):
            name, _ = cls.generate_name(genre=genre, **kwargs)
            names.append(name)
        return names
    
    @classmethod
    def _get_name_pool(cls, genre: TTRPGGenre) -> Dict[str, List[str]]:
        """Get the appropriate name pool for the genre."""
        genre_pools = {
            TTRPGGenre.FANTASY: cls.FANTASY_NAMES,
            TTRPGGenre.SCI_FI: cls.SCIFI_NAMES,
            TTRPGGenre.CYBERPUNK: cls.CYBERPUNK_NAMES,
            TTRPGGenre.COSMIC_HORROR: cls.COSMIC_HORROR_NAMES,
            TTRPGGenre.POST_APOCALYPTIC: cls.POST_APOCALYPTIC_NAMES,
            TTRPGGenre.WESTERN: cls.WESTERN_NAMES,
            TTRPGGenre.SUPERHERO: cls.SUPERHERO_NAMES,
            TTRPGGenre.SPACE_OPERA: cls.SCIFI_NAMES,  # Use sci-fi names
            TTRPGGenre.STEAMPUNK: cls.FANTASY_NAMES,  # Use fantasy with modifications
            TTRPGGenre.URBAN_FANTASY: cls.FANTASY_NAMES,  # Modern + fantasy mix
            TTRPGGenre.MODERN: cls.CYBERPUNK_NAMES,  # Modern names
        }
        return genre_pools.get(genre, cls.FANTASY_NAMES)
    
    @classmethod
    def _generate_first_name(
        cls,
        name_pool: Dict[str, List[str]],
        gender: str,
        race: Optional[CharacterRace]
    ) -> str:
        """Generate a first name based on gender and race."""
        # Get gender-appropriate names
        if gender == "male" and f"first_male" in name_pool:
            names = name_pool["first_male"]
        elif gender == "female" and f"first_female" in name_pool:
            names = name_pool["first_female"]
        else:
            names = name_pool.get("first_neutral", name_pool.get("first_male", []))
        
        if not names:
            names = ["Unknown"]
        
        name = random.choice(names)
        
        # Apply race modifications if applicable
        if race and race in cls.RACE_NAME_MODIFIERS:
            modifiers = cls.RACE_NAME_MODIFIERS[race]
            if "suffixes" in modifiers:
                name = name + random.choice(modifiers["suffixes"])
        
        return name
    
    @classmethod
    def _generate_last_name(
        cls,
        name_pool: Dict[str, List[str]],
        genre: TTRPGGenre,
        race: Optional[CharacterRace],
        role: Optional[NPCRole]
    ) -> Optional[str]:
        """Generate a last name based on context."""
        # Some genres/roles don't always use last names
        if genre == TTRPGGenre.POST_APOCALYPTIC and random.random() < 0.3:
            return None
        if role == NPCRole.CRIMINAL and random.random() < 0.4:
            return None
        
        # Determine which last name pool to use
        if role == NPCRole.NOBLE and "last_noble" in name_pool:
            names = name_pool["last_noble"]
        elif role in [NPCRole.COMMONER, NPCRole.MERCHANT] and "last_common" in name_pool:
            names = name_pool["last_common"]
        elif genre == TTRPGGenre.CYBERPUNK and role == NPCRole.CRIMINAL and "last_street" in name_pool:
            names = name_pool["last_street"]
        else:
            # Use the most appropriate default pool
            names = (name_pool.get("last_common") or 
                    name_pool.get("last_descriptive") or 
                    name_pool.get("last_colony") or
                    ["Smith", "Jones", "Williams"])
        
        return random.choice(names)
    
    @classmethod
    def _generate_title(
        cls,
        name_pool: Dict[str, List[str]],
        genre: TTRPGGenre,
        role: Optional[NPCRole],
        gender: str
    ) -> Optional[str]:
        """Generate an appropriate title."""
        if role and role in cls.NPC_ROLE_PATTERNS:
            role_titles = cls.NPC_ROLE_PATTERNS[role].get("titles", [])
            if role_titles:
                valid_titles = [t for t in role_titles if t]  # Filter empty strings
                if valid_titles:
                    title = random.choice(valid_titles)
                    if title:
                        return title
        
        if "titles" in name_pool:
            titles = name_pool["titles"]
            # Filter gender-appropriate titles if needed
            if gender == "female":
                female_titles = ["Lady", "Dame", "Mistress", "Baroness", "Mother", "Sister"]
                titles = [t for t in titles if t in female_titles or t in ["Doctor", "Professor", "Captain"]]
            elif gender == "male":
                male_titles = ["Sir", "Lord", "Master", "Baron", "Father", "Brother"]
                titles = [t for t in titles if t in male_titles or t in ["Doctor", "Professor", "Captain"]]
            
            if titles:
                return random.choice(titles)
        
        return None
    
    @classmethod
    def _generate_nickname(
        cls,
        name_pool: Dict[str, List[str]],
        genre: TTRPGGenre,
        race: Optional[CharacterRace],
        role: Optional[NPCRole]
    ) -> Optional[str]:
        """Generate a nickname or epithet."""
        # Cyberpunk uses handles
        if genre == TTRPGGenre.CYBERPUNK and "handles" in name_pool:
            if random.random() < 0.5:  # 50% chance for a handle
                return random.choice(name_pool["handles"])
        
        # Check for role-specific nicknames
        if role and role in cls.NPC_ROLE_PATTERNS:
            role_suffixes = cls.NPC_ROLE_PATTERNS[role].get("suffixes", [])
            if role_suffixes and random.random() < 0.3:
                suffix = random.choice(role_suffixes)
                # Replace placeholders
                suffix = suffix.replace("[Location]", random.choice(["Westmarch", "Eastwood", "Northshire"]))
                suffix = suffix.replace("[Ordinal]", random.choice(["First", "Second", "Third"]))
                suffix = suffix.replace("[Family]", random.choice(["Blackstone", "Goldshire", "Ravencrest"]))
                return suffix
        
        # Use genre-specific nicknames
        if "nicknames" in name_pool:
            return random.choice(name_pool["nicknames"])
        
        return None
    
    @classmethod
    def _apply_role_patterns(
        cls,
        role: NPCRole,
        title: Optional[str],
        nickname: Optional[str],
        genre: TTRPGGenre
    ) -> Tuple[Optional[str], Optional[str]]:
        """Apply role-specific naming patterns."""
        if role not in cls.NPC_ROLE_PATTERNS:
            return title, nickname
        
        patterns = cls.NPC_ROLE_PATTERNS[role]
        
        # Override with role-specific title if not already set
        if not title and patterns.get("titles"):
            valid_titles = [t for t in patterns["titles"] if t]
            if valid_titles:
                title = random.choice(valid_titles)
        
        # Add role-specific suffix as nickname if not already set
        if not nickname and patterns.get("suffixes"):
            if random.random() < 0.4:  # 40% chance
                nickname = random.choice(patterns["suffixes"])
        
        return title, nickname
    
    @classmethod
    def generate_organization_name(cls, genre: TTRPGGenre, org_type: str = "guild") -> str:
        """Generate names for organizations, guilds, and factions."""
        org_templates = {
            TTRPGGenre.FANTASY: {
                "guild": ["The {adj} {noun} Guild", "Order of the {adj} {noun}", "The {noun} Brotherhood"],
                "merchant": ["The {adj} Trading Company", "{noun} Merchants Consortium", "The {adj} Caravan"],
                "military": ["The {adj} Legion", "{noun} Regiment", "The {adj} Guard"],
                "religious": ["Church of the {adj} {noun}", "Temple of {noun}", "The {adj} Faith"],
            },
            TTRPGGenre.CYBERPUNK: {
                "corp": ["{adj} {noun} Corporation", "{noun} Industries", "{adj} Systems Inc."],
                "gang": ["The {adj} {noun}s", "{noun} Crew", "The {adj} Runners"],
                "hacker": ["{adj} {noun} Collective", "The {noun} Protocol", "{adj} Net"],
            },
            TTRPGGenre.SCI_FI: {
                "federation": ["The {adj} Federation", "{noun} Alliance", "United {noun} Systems"],
                "corporation": ["{adj} {noun} Dynamics", "{noun} Aerospace", "{adj} Industries"],
                "military": ["{noun} Fleet Command", "The {adj} Armada", "{noun} Defense Force"],
            },
            TTRPGGenre.POST_APOCALYPTIC: {
                "settlement": ["New {noun}", "{adj} Haven", "Fort {noun}"],
                "raider": ["The {adj} {noun}s", "{noun} Gang", "The {adj} Horde"],
                "faction": ["The {noun} Republic", "{adj} Brotherhood", "Order of {noun}"],
            },
        }
        
        adjectives = {
            TTRPGGenre.FANTASY: ["Silver", "Golden", "Iron", "Shadow", "Crystal", "Mystic", "Ancient"],
            TTRPGGenre.CYBERPUNK: ["Chrome", "Neon", "Digital", "Binary", "Quantum", "Neural", "Cyber"],
            TTRPGGenre.SCI_FI: ["Stellar", "Cosmic", "Galactic", "Quantum", "Nova", "Astral", "Void"],
            TTRPGGenre.POST_APOCALYPTIC: ["Rusted", "Broken", "Lost", "Last", "New", "Free", "United"],
        }
        
        nouns = {
            TTRPGGenre.FANTASY: ["Dragon", "Phoenix", "Sword", "Shield", "Crown", "Tower", "Rose"],
            TTRPGGenre.CYBERPUNK: ["Ghost", "Wire", "Data", "Chrome", "Matrix", "Grid", "Code"],
            TTRPGGenre.SCI_FI: ["Star", "Horizon", "Frontier", "Nebula", "Cosmos", "Empire", "Dawn"],
            TTRPGGenre.POST_APOCALYPTIC: ["Dawn", "Hope", "Steel", "Ash", "Storm", "Sun", "Earth"],
        }
        
        # Get templates for genre and type
        genre_templates = org_templates.get(genre, org_templates[TTRPGGenre.FANTASY])
        templates = genre_templates.get(org_type, genre_templates[list(genre_templates.keys())[0]])
        
        # Get adjectives and nouns
        adj_list = adjectives.get(genre, adjectives[TTRPGGenre.FANTASY])
        noun_list = nouns.get(genre, nouns[TTRPGGenre.FANTASY])
        
        # Generate name
        template = random.choice(templates)
        return template.format(
            adj=random.choice(adj_list),
            noun=random.choice(noun_list)
        )
    
    @classmethod
    def generate_location_name(cls, genre: TTRPGGenre, location_type: str = "city") -> str:
        """Generate names for locations based on genre and type."""
        location_templates = {
            TTRPGGenre.FANTASY: {
                "city": ["{adj} {noun}", "{noun}haven", "{noun}shire", "Port {noun}"],
                "town": ["{noun}ton", "{adj}brook", "{noun}dale", "{noun} Crossing"],
                "dungeon": ["The {adj} Depths", "{noun} Caverns", "The {adj} {noun}"],
                "fortress": ["Fort {noun}", "{adj} Keep", "Castle {noun}", "{noun} Citadel"],
            },
            TTRPGGenre.CYBERPUNK: {
                "city": ["Neo-{noun}", "{adj} City", "New {noun}", "Mega-{noun}"],
                "district": ["The {adj} Zone", "{noun} Sector", "{adj} Quarter", "Downtown {noun}"],
                "club": ["The {adj} {noun}", "Club {noun}", "{noun} Lounge", "The {adj}"],
            },
            TTRPGGenre.SCI_FI: {
                "planet": ["{noun} Prime", "{adj} {noun}", "New {noun}", "{noun}-{number}"],
                "station": ["{noun} Station", "{adj} Outpost", "Port {noun}", "{noun} Base"],
                "system": ["The {noun} System", "{adj} Sector", "{noun} Cluster", "{adj} Reach"],
            },
            TTRPGGenre.POST_APOCALYPTIC: {
                "settlement": ["New {noun}", "{adj}town", "Fort {noun}", "{noun} Falls"],
                "ruins": ["Old {noun}", "The {adj} Ruins", "{noun} Wasteland", "Dead {noun}"],
                "landmark": ["The {adj} {noun}", "{noun} Crater", "{adj} Zone", "Ground {noun}"],
            },
        }
        
        adj_pools = {
            TTRPGGenre.FANTASY: ["Silver", "Golden", "Dark", "Bright", "Ancient", "Hidden", "Lost"],
            TTRPGGenre.CYBERPUNK: ["Neon", "Chrome", "Digital", "Dark", "Upper", "Lower", "Central"],
            TTRPGGenre.SCI_FI: ["Alpha", "Beta", "Gamma", "New", "Far", "Deep", "Outer"],
            TTRPGGenre.POST_APOCALYPTIC: ["New", "Old", "Lost", "Dead", "Rusted", "Broken", "Last"],
        }
        
        noun_pools = {
            TTRPGGenre.FANTASY: ["Moon", "Star", "River", "Mountain", "Forest", "Stone", "Dragon"],
            TTRPGGenre.CYBERPUNK: ["Tokyo", "Berlin", "Angeles", "Hong Kong", "Moscow", "Cairo"],
            TTRPGGenre.SCI_FI: ["Terra", "Eden", "Haven", "Frontier", "Horizon", "Atlas", "Olympus"],
            TTRPGGenre.POST_APOCALYPTIC: ["Vegas", "York", "Hope", "Dawn", "Eden", "Haven", "Oasis"],
        }
        
        # Get appropriate pools
        templates = location_templates.get(genre, location_templates[TTRPGGenre.FANTASY])
        location_temps = templates.get(location_type, templates[list(templates.keys())[0]])
        
        adjs = adj_pools.get(genre, adj_pools[TTRPGGenre.FANTASY])
        nouns = noun_pools.get(genre, noun_pools[TTRPGGenre.FANTASY])
        
        # Generate name
        template = random.choice(location_temps)
        name = template.format(
            adj=random.choice(adjs),
            noun=random.choice(nouns),
            number=random.randint(1, 999)
        )
        
        return name