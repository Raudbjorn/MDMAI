"""Character and NPC generation module for TTRPG Assistant."""

from .character_generator import CharacterGenerator
from .backstory_generator import BackstoryGenerator
from .npc_generator import NPCGenerator
from .models import (
    Character,
    CharacterStats,
    CharacterClass,
    CharacterRace,
    Equipment,
    Backstory,
    NPC,
    NPCRole,
    PersonalityTrait
)
from .mcp_tools import (
    initialize_character_tools,
    register_character_tools
)

__all__ = [
    'CharacterGenerator',
    'BackstoryGenerator',
    'NPCGenerator',
    'Character',
    'CharacterStats',
    'CharacterClass',
    'CharacterRace',
    'Equipment',
    'Backstory',
    'NPC',
    'NPCRole',
    'PersonalityTrait',
    'initialize_character_tools',
    'register_character_tools'
]