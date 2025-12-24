"""
Genre classification logic for TTRPG PDFs.

This module provides intelligent genre classification based on content analysis,
keywords, and pattern matching from PDF text.
"""

import re
from collections import Counter
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path

from .models import TTRPGGenre


class GenreClassifier:
    """Classify TTRPG content by genre using keyword analysis and pattern matching."""
    
    def __init__(self):
        """Initialize the genre classifier with keyword mappings."""
        self.genre_keywords = self._initialize_genre_keywords()
        self.genre_patterns = self._initialize_genre_patterns()
        self.title_indicators = self._initialize_title_indicators()
        
    def _initialize_genre_keywords(self) -> Dict[TTRPGGenre, Set[str]]:
        """Initialize keyword sets for each genre."""
        return {
            TTRPGGenre.FANTASY: {
                "magic", "wizard", "sorcerer", "spell", "dragon", "elf", "dwarf", 
                "orc", "goblin", "sword", "shield", "armor", "knight", "paladin",
                "cleric", "rogue", "bard", "ranger", "druid", "necromancer",
                "dungeon", "castle", "kingdom", "quest", "tavern", "enchanted",
                "magical", "arcane", "divine", "potion", "scroll", "staff",
                "wand", "tome", "grimoire", "ritual", "summoning", "elemental",
                "mana", "mystic", "mythical", "legendary", "artifact", "relic"
            },
            
            TTRPGGenre.SCI_FI: {
                "spaceship", "laser", "plasma", "alien", "planet", "galaxy",
                "starship", "hyperspace", "android", "robot", "cyborg", "AI",
                "quantum", "neutron", "photon", "reactor", "fusion", "warp",
                "teleport", "blaster", "scanner", "sensor", "force field",
                "antimatter", "singularity", "terraforming", "xenobiology",
                "exoplanet", "stellar", "cosmic", "orbital", "zero-g", "vacuum",
                "radiation", "nanotech", "biotech", "genetic", "clone"
            },
            
            TTRPGGenre.CYBERPUNK: {
                "cybernetic", "hacker", "netrunner", "megacorp", "chrome",
                "augmentation", "implant", "neural", "matrix", "cyberspace",
                "data jack", "ICE", "black ICE", "street samurai", "fixer",
                "corpo", "edgerunner", "braindance", "daemon", "encryption",
                "firewall", "malware", "virus", "blockchain", "cryptocurrency",
                "surveillance", "dystopia", "neon", "sprawl", "arcology",
                "biotechnology", "wetware", "simstim", "deck", "console cowboy"
            },
            
            TTRPGGenre.COSMIC_HORROR: {
                "eldritch", "cosmic", "madness", "sanity", "tentacle", "void",
                "ancient", "forbidden", "unspeakable", "otherworldly", "nightmare",
                "cultist", "ritual", "summoning", "old ones", "great old one",
                "shoggoth", "deep ones", "mythos", "non-euclidean", "blasphemous",
                "cyclopean", "nameless", "indescribable", "terror", "dread",
                "corruption", "taint", "whispers", "shadows", "abyss", "beyond",
                "unknowable", "incomprehensible", "star spawn", "outer god"
            },
            
            TTRPGGenre.POST_APOCALYPTIC: {
                "wasteland", "survivor", "scavenger", "radiation", "mutant",
                "bunker", "vault", "fallout", "ruins", "raider", "apocalypse",
                "nuclear", "contamination", "hazmat", "geiger", "rad", "scrap",
                "salvage", "barter", "bottlecaps", "caravan", "settlement",
                "outbreak", "pandemic", "collapse", "anarchy", "warlord",
                "gasoline", "ammunition", "rations", "purifier", "shelter"
            },
            
            TTRPGGenre.STEAMPUNK: {
                "steam", "clockwork", "gear", "cog", "brass", "copper",
                "airship", "dirigible", "zeppelin", "automaton", "difference engine",
                "victorian", "industrial", "boiler", "pressure", "valve",
                "goggles", "corset", "top hat", "mechanical", "pneumatic",
                "tesla", "aether", "phlogiston", "analytical", "locomotive",
                "factory", "workshop", "inventor", "engineer", "tinkerer"
            },
            
            TTRPGGenre.URBAN_FANTASY: {
                "vampire", "werewolf", "fae", "witch", "supernatural", "occult",
                "modern magic", "hidden world", "masquerade", "covenant", "pack",
                "coven", "hunter", "slayer", "demon", "angel", "spirit",
                "ghost", "poltergeist", "possession", "exorcism", "ritual",
                "ward", "sanctuary", "threshold", "veil", "mundane", "awakened",
                "nightclub", "subway", "skyscraper", "alley", "urban"
            },
            
            TTRPGGenre.SPACE_OPERA: {
                "empire", "federation", "alliance", "fleet", "admiral", "captain",
                "starfighter", "battlecruiser", "dreadnought", "hyperdrive",
                "jump gate", "space station", "colony", "terraforming", "diplomat",
                "ambassador", "senate", "emperor", "rebellion", "resistance",
                "smuggler", "bounty hunter", "merchant", "trader", "pirate",
                "sector", "quadrant", "parsec", "light year", "nebula"
            },
            
            TTRPGGenre.SUPERHERO: {
                "superhero", "supervillain", "power", "mutation", "vigilante",
                "secret identity", "origin", "nemesis", "sidekick", "costume",
                "cape", "mask", "headquarters", "lair", "justice", "hero",
                "villain", "team", "league", "society", "ultra", "mega",
                "super strength", "flight", "invulnerability", "telepathy",
                "telekinesis", "energy blast", "transformation", "gadget"
            },
            
            TTRPGGenre.WESTERN: {
                "gunslinger", "sheriff", "outlaw", "saloon", "frontier",
                "cowboy", "ranch", "cattle", "horse", "revolver", "rifle",
                "duel", "showdown", "bounty", "wanted", "marshal", "deputy",
                "prospector", "gold rush", "mine", "railroad", "stagecoach",
                "prairie", "desert", "canyon", "mesa", "homestead", "bandito",
                "rustler", "poker", "whiskey", "boots", "spurs", "six-shooter"
            },
            
            TTRPGGenre.NOIR: {
                "detective", "investigator", "private eye", "case", "mystery",
                "murder", "crime", "evidence", "suspect", "witness", "alibi",
                "motive", "corruption", "dame", "femme fatale", "informant",
                "nightclub", "jazz", "cigarette", "fedora", "trench coat",
                "shadow", "rain", "neon", "alley", "office", "precinct",
                "commissioner", "mob", "gangster", "racket", "blackmail"
            },
            
            TTRPGGenre.HORROR: {
                "horror", "terror", "fear", "nightmare", "monster", "creature",
                "undead", "zombie", "skeleton", "ghost", "haunted", "curse",
                "hex", "evil", "darkness", "shadow", "blood", "gore", "scream",
                "death", "grave", "cemetery", "crypt", "tomb", "sacrifice",
                "ritual", "possessed", "demon", "hell", "abyss", "torment"
            },
            
            TTRPGGenre.MILITARY: {
                "soldier", "marine", "army", "navy", "air force", "squad",
                "platoon", "company", "battalion", "regiment", "division",
                "mission", "operation", "tactical", "strategic", "combat",
                "warfare", "weapon", "ammunition", "grenade", "explosive",
                "tank", "helicopter", "fighter", "bomber", "artillery",
                "sniper", "recon", "intel", "command", "officer", "sergeant"
            }
        }
    
    def _initialize_genre_patterns(self) -> Dict[TTRPGGenre, List[re.Pattern]]:
        """Initialize regex patterns for genre detection."""
        return {
            TTRPGGenre.FANTASY: [
                re.compile(r'\b\d+d\d+\b', re.IGNORECASE),  # Dice notation
                re.compile(r'\bspell\s?slot', re.IGNORECASE),
                re.compile(r'\bcantrip', re.IGNORECASE),
                re.compile(r'\bhit\s?points?\b', re.IGNORECASE),
            ],
            TTRPGGenre.CYBERPUNK: [
                re.compile(r'\b2\d{3}\b'),  # Year 20XX
                re.compile(r'\bcyber\w+', re.IGNORECASE),
                re.compile(r'\bnet\s?run', re.IGNORECASE),
                re.compile(r'\bcorp(?:oration)?', re.IGNORECASE),
            ],
            TTRPGGenre.COSMIC_HORROR: [
                re.compile(r'\bsan(?:ity)?\s?(?:check|loss)', re.IGNORECASE),
                re.compile(r'\bcthulhu', re.IGNORECASE),
                re.compile(r'\bold\s+ones?', re.IGNORECASE),
                re.compile(r'\beldritch\s+\w+', re.IGNORECASE),
            ],
            TTRPGGenre.POST_APOCALYPTIC: [
                re.compile(r'\brad(?:iation)?\s+(?:level|zone)', re.IGNORECASE),
                re.compile(r'\bvault\s+\d+', re.IGNORECASE),
                re.compile(r'\bwasteland', re.IGNORECASE),
            ],
        }
    
    def _initialize_title_indicators(self) -> Dict[TTRPGGenre, Set[str]]:
        """Initialize title-based indicators for genres."""
        return {
            TTRPGGenre.FANTASY: {"D&D", "Dungeons", "Dragons", "Pathfinder", "Fantasy"},
            TTRPGGenre.CYBERPUNK: {"Cyberpunk", "2020", "2077", "Shadowrun", "Interface"},
            TTRPGGenre.COSMIC_HORROR: {"Cthulhu", "Lovecraft", "Horror", "Mythos", "Delta Green"},
            TTRPGGenre.POST_APOCALYPTIC: {"Apocalypse", "Fallout", "Wasteland", "After"},
            TTRPGGenre.SPACE_OPERA: {"Star", "Space", "Galaxy", "Traveller", "Stars Without Number"},
            TTRPGGenre.WESTERN: {"West", "Western", "Deadlands", "Boot Hill", "Gunslinger"},
            TTRPGGenre.NOIR: {"Noir", "Detective", "Mystery", "Gumshoe", "City of Mist"},
            TTRPGGenre.STEAMPUNK: {"Steam", "Clockwork", "Victorian", "Brass", "Gear"},
            TTRPGGenre.SUPERHERO: {"Heroes", "Super", "Powers", "Mutants", "Masks"},
            TTRPGGenre.MILITARY: {"Ops", "Military", "Soldier", "Combat", "War", "Tactical"},
        }
    
    def classify_by_title(self, title: str) -> Optional[TTRPGGenre]:
        """Attempt to classify genre based on PDF title alone."""
        title_lower = title.lower()
        
        # Check for specific title indicators
        for genre, indicators in self.title_indicators.items():
            for indicator in indicators:
                if indicator.lower() in title_lower:
                    return genre
        
        # Check for year patterns suggesting cyberpunk/sci-fi
        if re.search(r'20\d{2}', title):
            if any(word in title_lower for word in ["cyber", "punk", "chrome", "net"]):
                return TTRPGGenre.CYBERPUNK
            elif int(re.search(r'20\d{2}', title).group()) > 2030:
                return TTRPGGenre.SCI_FI
        
        # Check for numbered series suggesting specific genres
        if re.search(r'\d{4}', title):
            if "xx" in title_lower or "18" in title:
                return TTRPGGenre.WESTERN
            elif "23" in title or "24" in title:
                return TTRPGGenre.CYBERPUNK
        
        return None
    
    def classify_by_content(self, text: str, sample_size: int = 5000) -> Tuple[TTRPGGenre, float]:
        """
        Classify genre based on text content analysis.
        
        Args:
            text: The text content to analyze
            sample_size: Number of characters to sample for analysis
        
        Returns:
            Tuple of (detected genre, confidence score)
        """
        # Sample text if too long
        if len(text) > sample_size:
            # Sample from beginning, middle, and end
            parts = [
                text[:sample_size // 3],
                text[len(text) // 2 - sample_size // 6:len(text) // 2 + sample_size // 6],
                text[-sample_size // 3:]
            ]
            text_sample = " ".join(parts)
        else:
            text_sample = text
        
        text_lower = text_sample.lower()
        genre_scores: Dict[TTRPGGenre, float] = Counter()
        
        # Score based on keyword matches
        for genre, keywords in self.genre_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Weight by keyword length (longer = more specific)
                    genre_scores[genre] += len(keyword) / 10
        
        # Score based on pattern matches
        for genre, patterns in self.genre_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text_sample)
                if matches:
                    genre_scores[genre] += len(matches) * 2
        
        # Check for genre combinations
        if genre_scores.get(TTRPGGenre.URBAN_FANTASY, 0) > 0 and \
           genre_scores.get(TTRPGGenre.FANTASY, 0) > 0:
            genre_scores[TTRPGGenre.URBAN_FANTASY] *= 1.5
        
        if genre_scores.get(TTRPGGenre.COSMIC_HORROR, 0) > 0 and \
           genre_scores.get(TTRPGGenre.HORROR, 0) > 0:
            genre_scores[TTRPGGenre.COSMIC_HORROR] *= 1.5
        
        # Get the top genre
        if genre_scores:
            top_genre = max(genre_scores, key=genre_scores.get)
            total_score = sum(genre_scores.values())
            confidence = genre_scores[top_genre] / total_score if total_score > 0 else 0
            return top_genre, confidence
        
        return TTRPGGenre.GENERIC, 0.0
    
    def classify(self, pdf_path: Path, title: Optional[str] = None, 
                 text_content: Optional[str] = None) -> Tuple[TTRPGGenre, float]:
        """
        Classify a PDF's genre using multiple strategies.
        
        Args:
            pdf_path: Path to the PDF file
            title: Optional title override
            text_content: Optional text content for analysis
        
        Returns:
            Tuple of (detected genre, confidence score)
        """
        # Try title-based classification first
        if title is None:
            title = pdf_path.stem
        
        title_genre = self.classify_by_title(title)
        
        # If we have text content, do content analysis
        if text_content:
            content_genre, confidence = self.classify_by_content(text_content)
            
            # If title and content agree, boost confidence
            if title_genre and title_genre == content_genre:
                return content_genre, min(confidence * 1.2, 1.0)
            
            # If title genre exists but differs, use content with reduced confidence
            elif title_genre and confidence < 0.7:
                return title_genre, 0.6
            
            return content_genre, confidence
        
        # If only title classification available
        if title_genre:
            return title_genre, 0.7
        
        # Default fallback
        return TTRPGGenre.GENERIC, 0.3
    
    def get_genre_description(self, genre: TTRPGGenre) -> str:
        """Get a human-readable description of a genre."""
        descriptions = {
            TTRPGGenre.FANTASY: "Classic fantasy with magic, medieval settings, and mythical creatures",
            TTRPGGenre.SCI_FI: "Science fiction with advanced technology and space exploration",
            TTRPGGenre.CYBERPUNK: "Near-future dystopia with cybernetics and corporate dominance",
            TTRPGGenre.COSMIC_HORROR: "Lovecraftian horror with ancient entities and sanity mechanics",
            TTRPGGenre.POST_APOCALYPTIC: "Survival in a world after catastrophic collapse",
            TTRPGGenre.STEAMPUNK: "Victorian-era technology with steam-powered innovations",
            TTRPGGenre.URBAN_FANTASY: "Modern world with hidden supernatural elements",
            TTRPGGenre.SPACE_OPERA: "Epic space adventures with galactic civilizations",
            TTRPGGenre.SUPERHERO: "Comic book style heroes with extraordinary abilities",
            TTRPGGenre.WESTERN: "Old West frontier settings with gunfights and outlaws",
            TTRPGGenre.NOIR: "Dark detective stories with mystery and moral ambiguity",
            TTRPGGenre.HORROR: "Frightening scenarios with monsters and survival",
            TTRPGGenre.MILITARY: "Modern or historical military operations and tactics",
            TTRPGGenre.PULP: "Adventure serials with larger-than-life heroes",
            TTRPGGenre.HISTORICAL: "Real historical periods as game settings",
            TTRPGGenre.MYTHOLOGICAL: "Ancient myths and legends brought to life",
            TTRPGGenre.ANIME: "Japanese animation inspired settings and tropes",
            TTRPGGenre.MODERN: "Contemporary real-world settings",
            TTRPGGenre.MYSTERY: "Investigation and puzzle-solving focused gameplay",
            TTRPGGenre.GENERIC: "System-agnostic or multi-genre content",
            TTRPGGenre.UNKNOWN: "Genre could not be determined",
        }
        return descriptions.get(genre, "Unknown genre type")