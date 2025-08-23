"""Character and NPC generation module for TTRPG Assistant."""

from .backstory_generator import BackstoryGenerator
from .character_generator import CharacterGenerator
from .mcp_tools import initialize_character_tools, register_character_tools
from .models import (
    NPC,
    Backstory,
    Character,
    CharacterClass,
    CharacterRace,
    CharacterStats,
    Equipment,
    NPCRole,
    PersonalityTrait,
)
from .npc_generator import NPCGenerator

__all__ = [
    "CharacterGenerator",
    "BackstoryGenerator",
    "NPCGenerator",
    "Character",
    "CharacterStats",
    "CharacterClass",
    "CharacterRace",
    "Equipment",
    "Backstory",
    "NPC",
    "NPCRole",
    "PersonalityTrait",
    "initialize_character_tools",
    "register_character_tools",
]
