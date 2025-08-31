"""
Pattern-based content extraction for TTRPG PDFs.

This module provides sophisticated pattern matching and NLP techniques
to extract game elements from TTRPG rulebooks.
"""

import re
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import json

from .models import (
    ExtendedCharacterRace, ExtendedCharacterClass, ExtendedNPCRole,
    ExtendedEquipment, ContentType, ExtractionConfidence, SourceAttribution,
    TTRPGGenre
)


class ContentExtractor:
    """Extract TTRPG content using pattern matching and NLP techniques."""
    
    def __init__(self, genre: TTRPGGenre = TTRPGGenre.GENERIC):
        """Initialize the content extractor for a specific genre."""
        self.genre = genre
        self.patterns = self._initialize_patterns()
        self.section_headers = self._initialize_section_headers()
        
    def _initialize_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Initialize regex patterns for content extraction."""
        return {
            "race": [
                re.compile(r'(?:Race|Species|Ancestry|Lineage|Heritage):\s*(\w+)', re.IGNORECASE),
                re.compile(r'(\w+)\s+(?:Traits|Racial Traits|Features)', re.IGNORECASE),
                re.compile(r'Playing (?:a|an) (\w+)', re.IGNORECASE),
                re.compile(r'(\w+) (?:get|gain|have) the following', re.IGNORECASE),
            ],
            "class": [
                re.compile(r'(?:Class|Profession|Career|Role):\s*(\w+)', re.IGNORECASE),
                re.compile(r'The (\w+) (?:Class|Profession)', re.IGNORECASE),
                re.compile(r'(\w+) (?:Class Features|Abilities)', re.IGNORECASE),
                re.compile(r'(?:Starting|Beginning) as (?:a|an) (\w+)', re.IGNORECASE),
            ],
            "npc": [
                re.compile(r'(\w+)\s+CR\s*\d+', re.IGNORECASE),
                re.compile(r'(\w+)\s+(?:Challenge|Threat)\s+(?:Rating|Level)', re.IGNORECASE),
                re.compile(r'(?:Named|Notable|Important)\s+NPCs?:\s*(\w+)', re.IGNORECASE),
            ],
            "equipment": [
                re.compile(r'(\w+(?:\s+\w+)?)\s+(?:Damage|DMG):\s*\d+d\d+', re.IGNORECASE),
                re.compile(r'(\w+(?:\s+\w+)?)\s+(?:AC|Armor Class):\s*\d+', re.IGNORECASE),
                re.compile(r'(\w+(?:\s+\w+)?)\s+(?:Cost|Price):\s*\d+', re.IGNORECASE),
                re.compile(r'(?:Weapon|Armor|Item|Equipment):\s*(\w+(?:\s+\w+)?)', re.IGNORECASE),
            ],
            "stat_block": [
                re.compile(r'(?:STR|Strength)[:\s]+(\d+)', re.IGNORECASE),
                re.compile(r'(?:DEX|Dexterity)[:\s]+(\d+)', re.IGNORECASE),
                re.compile(r'(?:CON|Constitution)[:\s]+(\d+)', re.IGNORECASE),
                re.compile(r'(?:INT|Intelligence)[:\s]+(\d+)', re.IGNORECASE),
                re.compile(r'(?:WIS|Wisdom)[:\s]+(\d+)', re.IGNORECASE),
                re.compile(r'(?:CHA|Charisma)[:\s]+(\d+)', re.IGNORECASE),
            ],
            "dice": [
                re.compile(r'\b(\d+)d(\d+)(?:\+(\d+))?\b'),
            ],
            "level": [
                re.compile(r'(?:Level|Lvl|Lv)[.\s]+(\d+)', re.IGNORECASE),
                re.compile(r'(\d+)(?:st|nd|rd|th)\s+(?:Level|level)', re.IGNORECASE),
            ],
        }
    
    def _initialize_section_headers(self) -> Dict[str, List[str]]:
        """Initialize common section headers by content type."""
        return {
            "races": [
                "Races", "Species", "Ancestries", "Lineages", "Heritages",
                "Character Races", "Playable Races", "Player Species"
            ],
            "classes": [
                "Classes", "Professions", "Careers", "Roles", "Archetypes",
                "Character Classes", "Player Classes", "Class Options"
            ],
            "npcs": [
                "NPCs", "Non-Player Characters", "Adversaries", "Enemies",
                "Monsters", "Creatures", "Bestiary", "Encounters"
            ],
            "equipment": [
                "Equipment", "Gear", "Items", "Weapons", "Armor", "Tools",
                "Inventory", "Shop", "Store", "Market", "Armory"
            ],
            "spells": [
                "Spells", "Magic", "Powers", "Abilities", "Techniques",
                "Psionics", "Miracles", "Invocations", "Rituals"
            ],
        }
    
    def extract_races(self, text: str, page_num: int, source: SourceAttribution) -> List[ExtendedCharacterRace]:
        """Extract character races from text."""
        races = []
        
        # Find race sections
        race_sections = self._find_sections(text, self.section_headers["races"])
        
        for section in race_sections:
            # Extract individual races
            race_matches = []
            for pattern in self.patterns["race"]:
                race_matches.extend(pattern.findall(section))
            
            for match in set(race_matches):
                race_name = match if isinstance(match, str) else match[0]
                
                # Extract race details
                race_text = self._extract_context(section, race_name, lines=20)
                
                # Parse race attributes
                race = ExtendedCharacterRace(
                    name=race_name.title(),
                    genre=self.genre,
                    description=self._extract_description(race_text),
                    traits=self._extract_traits(race_text),
                    abilities=self._extract_abilities(race_text),
                    stat_modifiers=self._extract_stat_modifiers(race_text),
                    size=self._extract_size(race_text),
                    speed=self._extract_speed(race_text),
                    languages=self._extract_languages(race_text),
                    source=source
                )
                
                # Add genre-specific tags
                race.tags = self._generate_tags(race_name, "race")
                races.append(race)
        
        return races
    
    def extract_classes(self, text: str, page_num: int, source: SourceAttribution) -> List[ExtendedCharacterClass]:
        """Extract character classes from text."""
        classes = []
        
        # Find class sections
        class_sections = self._find_sections(text, self.section_headers["classes"])
        
        for section in class_sections:
            # Extract individual classes
            class_matches = []
            for pattern in self.patterns["class"]:
                class_matches.extend(pattern.findall(section))
            
            for match in set(class_matches):
                class_name = match if isinstance(match, str) else match[0]
                
                # Extract class details
                class_text = self._extract_context(section, class_name, lines=30)
                
                # Parse class attributes
                char_class = ExtendedCharacterClass(
                    name=class_name.title(),
                    genre=self.genre,
                    description=self._extract_description(class_text),
                    hit_dice=self._extract_hit_dice(class_text),
                    primary_ability=self._extract_primary_ability(class_text),
                    saves=self._extract_saves(class_text),
                    skills=self._extract_skills(class_text),
                    equipment=self._extract_starting_equipment(class_text),
                    features=self._extract_class_features(class_text),
                    source=source
                )
                
                # Add genre-specific tags
                char_class.tags = self._generate_tags(class_name, "class")
                classes.append(char_class)
        
        return classes
    
    def extract_npcs(self, text: str, page_num: int, source: SourceAttribution) -> List[ExtendedNPCRole]:
        """Extract NPCs from text."""
        npcs = []
        
        # Find NPC sections
        npc_sections = self._find_sections(text, self.section_headers["npcs"])
        
        for section in npc_sections:
            # Extract individual NPCs
            npc_matches = []
            for pattern in self.patterns["npc"]:
                npc_matches.extend(pattern.findall(section))
            
            # Also look for stat blocks
            stat_blocks = self._find_stat_blocks(section)
            
            for npc_data in stat_blocks:
                npc = ExtendedNPCRole(
                    name=npc_data.get("name", "Unknown NPC"),
                    genre=self.genre,
                    description=npc_data.get("description", ""),
                    role_type=self._determine_npc_role(npc_data),
                    challenge_rating=npc_data.get("cr"),
                    abilities=npc_data.get("abilities", {}),
                    stats=npc_data.get("stats", {}),
                    skills=npc_data.get("skills", []),
                    traits=npc_data.get("traits", []),
                    actions=npc_data.get("actions", []),
                    source=source
                )
                
                # Add genre-specific tags
                npc.tags = self._generate_tags(npc.name, "npc")
                npcs.append(npc)
        
        return npcs
    
    def extract_equipment(self, text: str, page_num: int, source: SourceAttribution) -> List[ExtendedEquipment]:
        """Extract equipment from text."""
        equipment = []
        
        # Find equipment sections
        equipment_sections = self._find_sections(text, self.section_headers["equipment"])
        
        for section in equipment_sections:
            # Extract individual items
            item_matches = []
            for pattern in self.patterns["equipment"]:
                item_matches.extend(pattern.findall(section))
            
            for match in set(item_matches):
                item_name = match if isinstance(match, str) else match[0]
                
                # Extract item details
                item_text = self._extract_context(section, item_name, lines=10)
                
                # Parse equipment attributes
                item = ExtendedEquipment(
                    name=item_name.title(),
                    genre=self.genre,
                    equipment_type=self._determine_equipment_type(item_text),
                    description=self._extract_description(item_text),
                    cost=self._extract_cost(item_text),
                    weight=self._extract_weight(item_text),
                    properties=self._extract_properties(item_text),
                    damage=self._extract_damage(item_text),
                    armor_class=self._extract_armor_class(item_text),
                    source=source
                )
                
                # Add genre-specific attributes
                if self.genre in [TTRPGGenre.SCI_FI, TTRPGGenre.CYBERPUNK]:
                    item.tech_level = self._extract_tech_level(item_text)
                
                # Add tags
                item.tags = self._generate_tags(item_name, "equipment")
                equipment.append(item)
        
        return equipment
    
    def _find_sections(self, text: str, headers: List[str]) -> List[str]:
        """Find text sections based on headers."""
        sections = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            for header in headers:
                if header.lower() in line.lower():
                    # Extract section until next major header
                    section_lines = []
                    j = i + 1
                    while j < len(lines):
                        # Check if we hit another major section
                        if any(h.lower() in lines[j].lower() for h_list in self.section_headers.values() for h in h_list):
                            break
                        section_lines.append(lines[j])
                        j += 1
                    
                    if section_lines:
                        sections.append('\n'.join(section_lines))
        
        return sections
    
    def _extract_context(self, text: str, term: str, lines: int = 10) -> str:
        """Extract context around a term."""
        text_lines = text.split('\n')
        term_lower = term.lower()
        
        for i, line in enumerate(text_lines):
            if term_lower in line.lower():
                start = max(0, i - lines // 2)
                end = min(len(text_lines), i + lines // 2)
                return '\n'.join(text_lines[start:end])
        
        return ""
    
    def _extract_description(self, text: str) -> str:
        """Extract a description from text."""
        # Look for description patterns
        desc_patterns = [
            re.compile(r'Description:\s*(.+?)(?:\n\n|\n[A-Z])', re.DOTALL | re.IGNORECASE),
            re.compile(r'(?:^|\n)([A-Z].+?\.(?:\s+[A-Z].+?\.)*)(?:\n|$)', re.MULTILINE),
        ]
        
        for pattern in desc_patterns:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
        
        # Return first few sentences
        sentences = re.split(r'[.!?]+', text)
        if sentences:
            return '. '.join(sentences[:2]).strip() + '.'
        
        return "No description available."
    
    def _extract_traits(self, text: str) -> List[str]:
        """Extract traits or features from text."""
        traits = []
        
        # Look for trait patterns
        trait_patterns = [
            re.compile(r'(?:Trait|Feature|Ability):\s*(.+?)(?:\n|$)', re.IGNORECASE),
            re.compile(r'[•·▪]\s*(.+?)(?:\n|$)'),
            re.compile(r'(?:^|\n)\s*-\s*(.+?)(?:\n|$)'),
        ]
        
        for pattern in trait_patterns:
            matches = pattern.findall(text)
            traits.extend([m.strip() for m in matches if m.strip()])
        
        return list(set(traits))[:10]  # Limit to 10 unique traits
    
    def _extract_abilities(self, text: str) -> Dict[str, Any]:
        """Extract abilities from text."""
        abilities = {}
        
        # Look for ability patterns
        ability_patterns = [
            re.compile(r'(\w+):\s*\+?(\d+)'),
            re.compile(r'(\w+)\s+(?:Bonus|Modifier):\s*\+?(\d+)', re.IGNORECASE),
        ]
        
        for pattern in ability_patterns:
            matches = pattern.findall(text)
            for name, value in matches:
                abilities[name.title()] = int(value)
        
        return abilities
    
    def _extract_stat_modifiers(self, text: str) -> Dict[str, int]:
        """Extract stat modifiers from text."""
        modifiers = {}
        
        # Standard D&D-style stats
        stat_names = ["STR", "DEX", "CON", "INT", "WIS", "CHA",
                     "Strength", "Dexterity", "Constitution", 
                     "Intelligence", "Wisdom", "Charisma"]
        
        for stat in stat_names:
            pattern = re.compile(rf'{stat}[:\s]+\+?(\d+)', re.IGNORECASE)
            match = pattern.search(text)
            if match:
                key = stat[:3].upper() if len(stat) > 3 else stat.upper()
                modifiers[key] = int(match.group(1))
        
        return modifiers
    
    def _extract_size(self, text: str) -> str:
        """Extract size category from text."""
        sizes = ["Tiny", "Small", "Medium", "Large", "Huge", "Gargantuan", "Colossal"]
        text_lower = text.lower()
        
        for size in sizes:
            if size.lower() in text_lower:
                return size
        
        return "Medium"
    
    def _extract_speed(self, text: str) -> str:
        """Extract movement speed from text."""
        speed_patterns = [
            re.compile(r'Speed:?\s*(\d+)\s*(?:feet|ft|meters|m)', re.IGNORECASE),
            re.compile(r'(\d+)\s*(?:feet|ft|meters|m)\s+(?:per|/)\s+(?:round|turn|action)', re.IGNORECASE),
        ]
        
        for pattern in speed_patterns:
            match = pattern.search(text)
            if match:
                return f"{match.group(1)} ft"
        
        return "30 ft"
    
    def _extract_languages(self, text: str) -> List[str]:
        """Extract languages from text."""
        languages = []
        
        # Common language names
        common_languages = [
            "Common", "Elvish", "Dwarvish", "Orcish", "Goblin", "Draconic",
            "Celestial", "Abyssal", "Infernal", "Primordial", "Sylvan",
            "Undercommon", "Thieves' Cant", "Druidic", "Giant", "Gnomish"
        ]
        
        text_lower = text.lower()
        for lang in common_languages:
            if lang.lower() in text_lower:
                languages.append(lang)
        
        # Also look for language patterns
        lang_pattern = re.compile(r'Languages?:?\s*([^.\n]+)', re.IGNORECASE)
        match = lang_pattern.search(text)
        if match:
            lang_text = match.group(1)
            # Split by common delimiters
            parts = re.split(r'[,;]|\band\b', lang_text)
            for part in parts:
                clean = part.strip()
                if clean and len(clean) < 30:
                    languages.append(clean.title())
        
        return list(set(languages))
    
    def _extract_hit_dice(self, text: str) -> Optional[str]:
        """Extract hit dice from text."""
        patterns = [
            re.compile(r'Hit Dice?:?\s*(\d+d\d+)', re.IGNORECASE),
            re.compile(r'HD:?\s*(\d+d\d+)', re.IGNORECASE),
        ]
        
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_primary_ability(self, text: str) -> Optional[str]:
        """Extract primary ability from text."""
        patterns = [
            re.compile(r'Primary\s+(?:Ability|Stat|Attribute):?\s*(\w+)', re.IGNORECASE),
            re.compile(r'Key\s+(?:Ability|Stat|Attribute):?\s*(\w+)', re.IGNORECASE),
        ]
        
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return match.group(1).title()
        
        return None
    
    def _extract_saves(self, text: str) -> List[str]:
        """Extract saving throw proficiencies from text."""
        saves = []
        
        patterns = [
            re.compile(r'Saving Throws?:?\s*([^.\n]+)', re.IGNORECASE),
            re.compile(r'Saves?:?\s*([^.\n]+)', re.IGNORECASE),
        ]
        
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                save_text = match.group(1)
                # Extract individual saves
                parts = re.split(r'[,;&]|\band\b', save_text)
                for part in parts:
                    clean = part.strip()
                    if clean and len(clean) < 20:
                        saves.append(clean.title())
        
        return saves
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from text."""
        skills = []
        
        patterns = [
            re.compile(r'Skills?:?\s*([^.\n]+)', re.IGNORECASE),
            re.compile(r'Proficienc(?:y|ies):?\s*([^.\n]+)', re.IGNORECASE),
        ]
        
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                skill_text = match.group(1)
                # Extract individual skills
                parts = re.split(r'[,;]|\band\b', skill_text)
                for part in parts:
                    clean = part.strip()
                    if clean and len(clean) < 30:
                        skills.append(clean.title())
        
        return list(set(skills))
    
    def _extract_starting_equipment(self, text: str) -> List[str]:
        """Extract starting equipment from text."""
        equipment = []
        
        patterns = [
            re.compile(r'(?:Starting\s+)?Equipment:?\s*([^.\n]+)', re.IGNORECASE),
            re.compile(r'(?:Starting\s+)?Gear:?\s*([^.\n]+)', re.IGNORECASE),
        ]
        
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                equip_text = match.group(1)
                # Extract individual items
                parts = re.split(r'[,;]|\band\b|\bor\b', equip_text)
                for part in parts:
                    clean = part.strip()
                    if clean and len(clean) < 50:
                        equipment.append(clean)
        
        return equipment
    
    def _extract_class_features(self, text: str) -> Dict[int, List[str]]:
        """Extract class features by level."""
        features = defaultdict(list)
        
        # Look for level-based features
        level_pattern = re.compile(r'(?:Level|Lvl|Lv)[.\s]+(\d+)[:\s]+([^.\n]+)', re.IGNORECASE)
        matches = level_pattern.findall(text)
        
        for level, feature_text in matches:
            level_num = int(level)
            # Split multiple features
            parts = re.split(r'[,;]|\band\b', feature_text)
            for part in parts:
                clean = part.strip()
                if clean and len(clean) < 100:
                    features[level_num].append(clean)
        
        return dict(features)
    
    def _find_stat_blocks(self, text: str) -> List[Dict[str, Any]]:
        """Find and parse stat blocks in text."""
        stat_blocks = []
        
        # Look for stat block patterns
        lines = text.split('\n')
        current_block = {}
        in_block = False
        
        for line in lines:
            # Check for stat block start (has multiple stats on nearby lines)
            if any(pattern.search(line) for pattern in self.patterns["stat_block"]):
                in_block = True
                current_block = {"stats": {}}
            
            if in_block:
                # Extract stats
                for pattern in self.patterns["stat_block"]:
                    match = pattern.search(line)
                    if match:
                        stat_name = pattern.pattern.split('(?:')[1].split('|')[0][:3].upper()
                        current_block["stats"][stat_name] = int(match.group(1))
                
                # Extract name (usually at start of block)
                if "name" not in current_block and line.strip() and not any(c.isdigit() for c in line[:5]):
                    current_block["name"] = line.strip()
                
                # Extract CR
                cr_pattern = re.compile(r'CR\s*(\d+(?:/\d+)?)', re.IGNORECASE)
                cr_match = cr_pattern.search(line)
                if cr_match:
                    current_block["cr"] = cr_match.group(1)
                
                # Check for block end
                if len(current_block.get("stats", {})) >= 4 and line.strip() == "":
                    if current_block.get("name"):
                        stat_blocks.append(current_block)
                    current_block = {}
                    in_block = False
        
        return stat_blocks
    
    def _determine_npc_role(self, npc_data: Dict[str, Any]) -> str:
        """Determine NPC role type based on data."""
        cr = npc_data.get("cr", "")
        name = npc_data.get("name", "").lower()
        
        # Boss indicators
        if any(word in name for word in ["lord", "king", "queen", "master", "ancient", "elder"]):
            return "Boss"
        
        # Support indicators
        if any(word in name for word in ["healer", "cleric", "medic", "doctor"]):
            return "Support"
        
        # Social indicators
        if any(word in name for word in ["merchant", "noble", "diplomat", "spy"]):
            return "Social"
        
        # Default to combat
        return "Combat"
    
    def _determine_equipment_type(self, text: str) -> str:
        """Determine equipment type from text."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["damage", "attack", "weapon"]):
            return "weapon"
        elif any(word in text_lower for word in ["armor", "ac", "defense", "protection"]):
            return "armor"
        elif any(word in text_lower for word in ["tool", "kit", "supplies"]):
            return "tool"
        elif any(word in text_lower for word in ["potion", "consumable", "use"]):
            return "consumable"
        else:
            return "item"
    
    def _extract_cost(self, text: str) -> Optional[str]:
        """Extract cost from text."""
        patterns = [
            re.compile(r'(?:Cost|Price):?\s*([\d,]+)\s*(?:gp|gold|credits|¢|€)', re.IGNORECASE),
            re.compile(r'([\d,]+)\s*(?:gp|gold|credits|¢|€)', re.IGNORECASE),
        ]
        
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return match.group(0).strip()
        
        return None
    
    def _extract_weight(self, text: str) -> Optional[str]:
        """Extract weight from text."""
        patterns = [
            re.compile(r'Weight:?\s*([\d.]+)\s*(?:lb|lbs|kg|pounds)', re.IGNORECASE),
            re.compile(r'([\d.]+)\s*(?:lb|lbs|kg|pounds)', re.IGNORECASE),
        ]
        
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return match.group(0).strip()
        
        return None
    
    def _extract_properties(self, text: str) -> List[str]:
        """Extract item properties from text."""
        properties = []
        
        # Common property keywords
        property_keywords = [
            "finesse", "versatile", "two-handed", "light", "heavy", "reach",
            "thrown", "ammunition", "loading", "special", "silvered", "magical",
            "cursed", "sentient", "artifact", "legendary", "rare", "uncommon"
        ]
        
        text_lower = text.lower()
        for prop in property_keywords:
            if prop in text_lower:
                properties.append(prop.title())
        
        return properties
    
    def _extract_damage(self, text: str) -> Optional[str]:
        """Extract damage from text."""
        patterns = [
            re.compile(r'Damage:?\s*(\d+d\d+(?:\+\d+)?)', re.IGNORECASE),
            re.compile(r'(\d+d\d+(?:\+\d+)?)\s+damage', re.IGNORECASE),
        ]
        
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_armor_class(self, text: str) -> Optional[str]:
        """Extract armor class from text."""
        patterns = [
            re.compile(r'(?:AC|Armor Class):?\s*(\d+)', re.IGNORECASE),
            re.compile(r'\+(\d+)\s+(?:AC|armor)', re.IGNORECASE),
        ]
        
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_tech_level(self, text: str) -> Optional[str]:
        """Extract technology level for sci-fi items."""
        tech_levels = ["Primitive", "Industrial", "Information", "Fusion", 
                      "Antimatter", "Exotic", "Transcendent"]
        
        text_lower = text.lower()
        for level in tech_levels:
            if level.lower() in text_lower:
                return level
        
        # Check for TL notation
        tl_pattern = re.compile(r'TL[:\s]*(\d+)', re.IGNORECASE)
        match = tl_pattern.search(text)
        if match:
            return f"TL{match.group(1)}"
        
        return None
    
    def _generate_tags(self, name: str, content_type: str) -> Set[str]:
        """Generate tags for content based on name and type."""
        tags = {content_type, self.genre.name.lower()}
        
        # Add name-based tags
        name_lower = name.lower()
        
        # Combat tags
        if any(word in name_lower for word in ["warrior", "fighter", "soldier", "guard"]):
            tags.add("combat")
        
        # Magic tags
        if any(word in name_lower for word in ["mage", "wizard", "sorcerer", "magic"]):
            tags.add("magic")
        
        # Stealth tags
        if any(word in name_lower for word in ["rogue", "thief", "assassin", "spy"]):
            tags.add("stealth")
        
        # Tech tags
        if any(word in name_lower for word in ["cyber", "tech", "hacker", "digital"]):
            tags.add("technology")
        
        return tags