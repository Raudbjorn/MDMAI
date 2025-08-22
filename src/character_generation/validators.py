"""Validation utilities for character generation."""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import fields

from .models import (
    Character,
    NPC,
    CharacterClass,
    CharacterRace,
    NPCRole,
    CharacterStats
)

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class CharacterValidator:
    """Validates character and NPC data for consistency and correctness."""
    
    # Valid stat ranges
    MIN_STAT = 1
    MAX_STAT = 30  # D&D 5e maximum with magic items
    MIN_LEVEL = 1
    MAX_LEVEL = 20
    
    # Valid HP ranges by level (approximate)
    MIN_HP_PER_LEVEL = 4
    MAX_HP_PER_LEVEL = 15
    
    @classmethod
    def validate_stats(cls, stats: CharacterStats) -> List[str]:
        """
        Validate character statistics.
        
        Args:
            stats: CharacterStats object to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check stat ranges
        for stat_name in ['strength', 'dexterity', 'constitution', 
                         'intelligence', 'wisdom', 'charisma']:
            value = getattr(stats, stat_name)
            if not cls.MIN_STAT <= value <= cls.MAX_STAT:
                errors.append(
                    f"{stat_name.capitalize()} must be between "
                    f"{cls.MIN_STAT} and {cls.MAX_STAT}, got {value}"
                )
        
        # Check level
        if not cls.MIN_LEVEL <= stats.level <= cls.MAX_LEVEL:
            errors.append(
                f"Level must be between {cls.MIN_LEVEL} and {cls.MAX_LEVEL}, "
                f"got {stats.level}"
            )
        
        # Check HP
        min_hp = cls.MIN_HP_PER_LEVEL * stats.level
        max_hp = cls.MAX_HP_PER_LEVEL * stats.level
        if not min_hp <= stats.hit_points <= max_hp:
            logger.warning(
                f"HP {stats.hit_points} outside typical range "
                f"[{min_hp}, {max_hp}] for level {stats.level}"
            )
        
        # Check AC (typical range)
        if not 8 <= stats.armor_class <= 25:
            logger.warning(
                f"AC {stats.armor_class} outside typical range [8, 25]"
            )
        
        return errors
    
    @classmethod
    def validate_character(cls, character: Character) -> List[str]:
        """
        Validate a complete character.
        
        Args:
            character: Character object to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate basic fields
        if not character.name or not character.name.strip():
            errors.append("Character must have a name")
        
        if character.name and len(character.name) > 100:
            errors.append("Character name too long (max 100 characters)")
        
        # Validate stats
        if character.stats:
            stat_errors = cls.validate_stats(character.stats)
            errors.extend(stat_errors)
        else:
            errors.append("Character must have stats")
        
        # Validate class and race
        if character.character_class == CharacterClass.CUSTOM and not character.custom_class:
            errors.append("Custom class requires custom_class field")
        
        if character.race == CharacterRace.CUSTOM and not character.custom_race:
            errors.append("Custom race requires custom_race field")
        
        # Validate equipment
        if character.equipment:
            # Handle both single Equipment object and list of Equipment
            equipment_list = character.equipment if isinstance(character.equipment, list) else [character.equipment]
            for equip in equipment_list:
                if hasattr(equip, 'name') and not equip.name:
                    errors.append("Equipment must have a name")
                if hasattr(equip, 'quantity') and equip.quantity < 0:
                    errors.append(f"Equipment quantity cannot be negative: {equip.name}")
        
        # Validate skills (no duplicates)
        if character.skills:
            skill_set = set()
            for skill in character.skills:
                if skill in skill_set:
                    errors.append(f"Duplicate skill: {skill}")
                skill_set.add(skill)
        
        # Validate proficiency bonus if it exists
        if character.stats and hasattr(character, 'proficiency_bonus') and character.proficiency_bonus:
            expected_bonus = 2 + ((character.stats.level - 1) // 4)
            if character.proficiency_bonus != expected_bonus:
                logger.warning(
                    f"Proficiency bonus {character.proficiency_bonus} doesn't match "
                    f"expected {expected_bonus} for level {character.stats.level}"
                )
        
        return errors
    
    @classmethod
    def validate_npc(cls, npc: NPC) -> List[str]:
        """
        Validate an NPC.
        
        Args:
            npc: NPC object to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        # Start with character validation
        errors = cls.validate_character(npc)
        
        # Additional NPC-specific validation
        if not npc.role:
            errors.append("NPC must have a role")
        
        if npc.faction and len(npc.faction) > 100:
            errors.append("Faction name too long (max 100 characters)")
        
        # Validate personality traits
        if npc.personality_traits:
            for trait in npc.personality_traits:
                if not trait.category:
                    errors.append("Personality trait must have a category")
                if not trait.description:
                    errors.append("Personality trait must have a description")
        
        return errors
    
    @classmethod
    def sanitize_input(cls, data: Dict[str, Any], model_class: type) -> Dict[str, Any]:
        """
        Sanitize and validate input data for a model class.
        
        Args:
            data: Input data dictionary
            model_class: Target model class (Character or NPC)
            
        Returns:
            Sanitized data dictionary
            
        Raises:
            ValidationError: If data cannot be sanitized
        """
        sanitized = {}
        
        # Get field names from the model
        model_fields = {f.name for f in fields(model_class)}
        
        for key, value in data.items():
            # Skip unknown fields  
            if key not in model_fields:
                logger.debug(f"Skipping unknown field: {key}")
                continue
            
            # Sanitize strings
            if isinstance(value, str):
                # Strip whitespace and limit length
                value = value.strip()[:500]  # Max field length
                
                # Use a whitelist approach for names and simple text fields
                if key in ['name', 'custom_class', 'custom_race', 'faction', 'location', 'occupation']:
                    # Allow only alphanumeric, spaces, hyphens, apostrophes, and common punctuation
                    import re
                    cleaned = re.sub(r'[^a-zA-Z0-9\s\-\'.,!?]', '', value)
                    if cleaned != value:
                        logger.warning(f"Sanitized field {key}: removed special characters")
                        value = cleaned
                
                # For text fields that might contain story elements, be less restrictive
                # but still remove obvious HTML/script tags
                elif key in ['backstory', 'backstory_hints', 'description', 'secrets']:
                    import re
                    # Remove HTML tags and script elements
                    value = re.sub(r'<[^>]+>', '', value)
                    value = re.sub(r'(?i)(javascript:|onerror=|onclick=|onload=)', '', value)
                    
                    if value != value.strip()[:500]:
                        logger.debug(f"Sanitized field {key}: removed potential HTML/script content")
            
            # Sanitize numbers
            elif isinstance(value, (int, float)):
                # Ensure reasonable bounds
                if abs(value) > 1e6:
                    logger.warning(f"Extremely large value in field {key}: {value}")
                    value = 0
                sanitized[key] = value
                continue
            
            # Sanitize lists
            elif isinstance(value, list):
                # Limit list size
                if len(value) > 100:
                    logger.warning(f"List too long in field {key}, truncating")
                    value = value[:100]
            
            sanitized[key] = value
        
        return sanitized
    
    @classmethod
    def validate_generation_params(
        cls,
        level: int,
        system: str,
        character_class: Optional[str] = None,
        race: Optional[str] = None
    ) -> List[str]:
        """
        Validate character generation parameters.
        
        Args:
            level: Character level
            system: Game system
            character_class: Optional character class
            race: Optional character race
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate level
        if not cls.MIN_LEVEL <= level <= cls.MAX_LEVEL:
            errors.append(f"Level must be between {cls.MIN_LEVEL} and {cls.MAX_LEVEL}")
        
        # Validate system
        valid_systems = ["D&D 5e", "Pathfinder", "Call of Cthulhu", "Generic"]
        if system not in valid_systems:
            logger.warning(f"Unknown system: {system}, using Generic")
        
        # Validate class if provided
        if character_class is not None and character_class.strip():
            try:
                CharacterClass(character_class.strip().lower())
            except ValueError:
                if character_class.strip().lower() != "custom":
                    logger.info(f"Unknown class: {character_class}, treating as custom")
        
        # Validate race if provided
        if race is not None and race.strip():
            try:
                # Use underscore to match enum values
                CharacterRace(race.strip().lower().replace(' ', '_').replace('-', '_'))
            except ValueError:
                if race.strip().lower() != "custom":
                    logger.info(f"Unknown race: {race}, treating as custom")
        
        return errors
