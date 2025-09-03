#!/usr/bin/env python3
"""Test script for genre-specific NPC and equipment generation."""

import json
import logging
from typing import Dict, List

from src.character_generation.equipment_generator import EquipmentGenerator, EquipmentQuality
from src.character_generation.models import NPCRole, TTRPGGenre
from src.character_generation.name_generator import NameGenerator, NameStyle
from src.character_generation.npc_generator import NPCGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_name_generation():
    """Test name generation across different genres."""
    print("\n" + "="*60)
    print("TESTING NAME GENERATION")
    print("="*60)
    
    genres = [
        TTRPGGenre.FANTASY,
        TTRPGGenre.SCI_FI,
        TTRPGGenre.CYBERPUNK,
        TTRPGGenre.COSMIC_HORROR,
        TTRPGGenre.POST_APOCALYPTIC,
        TTRPGGenre.WESTERN,
        TTRPGGenre.SUPERHERO
    ]
    
    for genre in genres:
        print(f"\n{genre.value.upper()} Names:")
        print("-" * 40)
        
        # Generate different types of names
        for _ in range(3):
            name, components = NameGenerator.generate_name(
                genre=genre,
                gender=None,  # Random
                include_title=True,
                include_nickname=True
            )
            print(f"  • {name}")
        
        # Generate organization name
        org_name = NameGenerator.generate_organization_name(genre, "guild")
        print(f"  Organization: {org_name}")
        
        # Generate location name
        location_name = NameGenerator.generate_location_name(genre, "city")
        print(f"  Location: {location_name}")


def test_equipment_generation():
    """Test equipment generation across different genres."""
    print("\n" + "="*60)
    print("TESTING EQUIPMENT GENERATION")
    print("="*60)
    
    genres = [
        TTRPGGenre.FANTASY,
        TTRPGGenre.SCI_FI,
        TTRPGGenre.CYBERPUNK,
        TTRPGGenre.POST_APOCALYPTIC,
        TTRPGGenre.WESTERN,
        TTRPGGenre.SUPERHERO
    ]
    
    for genre in genres:
        print(f"\n{genre.value.upper()} Equipment:")
        print("-" * 40)
        
        # Generate equipment for different wealth levels
        for wealth_level in ["poor", "standard", "wealthy"]:
            equipment = EquipmentGenerator.generate_equipment(
                genre=genre,
                npc_role=NPCRole.ADVENTURER,
                level=5,
                wealth_level=wealth_level,
                include_magical=wealth_level == "wealthy"
            )
            
            print(f"  {wealth_level.capitalize()} Equipment:")
            if equipment.weapons:
                print(f"    Weapons: {', '.join(equipment.weapons[:2])}")
            if equipment.armor:
                print(f"    Armor: {', '.join(equipment.armor[:2])}")
            if equipment.items:
                print(f"    Items: {', '.join(equipment.items[:3])}")
            if equipment.magic_items:
                print(f"    Special: {', '.join(equipment.magic_items[:1])}")
            
            # Show currency
            currency_str = ", ".join([f"{v} {k}" for k, v in equipment.currency.items()])
            print(f"    Currency: {currency_str}")


def test_npc_generation():
    """Test full NPC generation with different genres."""
    print("\n" + "="*60)
    print("TESTING NPC GENERATION")
    print("="*60)
    
    generator = NPCGenerator()
    
    test_cases = [
        {
            "system": "D&D 5e",
            "genre": TTRPGGenre.FANTASY,
            "role": "merchant",
            "importance": "supporting"
        },
        {
            "system": "Cyberpunk 2020",
            "genre": TTRPGGenre.CYBERPUNK,
            "role": "criminal",
            "importance": "major"
        },
        {
            "system": "Traveller",
            "genre": TTRPGGenre.SCI_FI,
            "role": "guard",
            "importance": "minor"
        },
        {
            "system": "Call of Cthulhu",
            "genre": TTRPGGenre.COSMIC_HORROR,
            "role": "scholar",
            "importance": "major"
        },
        {
            "system": "Fallout",
            "genre": TTRPGGenre.POST_APOCALYPTIC,
            "role": "merchant",
            "importance": "supporting"
        },
        {
            "system": "Deadlands",
            "genre": TTRPGGenre.WESTERN,
            "role": "criminal",
            "importance": "major"
        },
        {
            "system": "Masks",
            "genre": TTRPGGenre.SUPERHERO,
            "role": "noble",
            "importance": "supporting"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['genre'].value.upper()} NPC ({test_case['system']}):")
        print("-" * 40)
        
        try:
            npc = generator.generate_npc(
                system=test_case["system"],
                role=test_case["role"],
                importance=test_case["importance"],
                level=5,
                genre=test_case["genre"],
                backstory_depth="simple"
            )
            
            print(f"  Name: {npc.name}")
            print(f"  Role: {npc.get_role_name()}")
            print(f"  Importance: {npc.importance}")
            
            if npc.stats:
                print(f"  Stats: STR {npc.stats.strength}, DEX {npc.stats.dexterity}, "
                      f"CON {npc.stats.constitution}")
            
            if npc.equipment:
                if npc.equipment.weapons:
                    print(f"  Weapons: {', '.join(npc.equipment.weapons[:2])}")
                if npc.equipment.armor:
                    print(f"  Armor: {', '.join(npc.equipment.armor[:2])}")
            
            if npc.personality_traits:
                traits = [t.trait for t in npc.personality_traits[:2]]
                print(f"  Personality: {', '.join(traits)}")
            
            if npc.secrets:
                print(f"  Secret: {npc.secrets[0]}")
                
        except Exception as e:
            print(f"  ERROR: {str(e)}")


def test_genre_specific_features():
    """Test genre-specific features and variations."""
    print("\n" + "="*60)
    print("TESTING GENRE-SPECIFIC FEATURES")
    print("="*60)
    
    # Test cyberpunk-specific generation
    print("\nCyberpunk Netrunner:")
    print("-" * 40)
    
    name, _ = NameGenerator.generate_name(
        genre=TTRPGGenre.CYBERPUNK,
        style=NameStyle.ALIAS,
        include_nickname=True
    )
    print(f"  Handle: {name}")
    
    equipment = EquipmentGenerator.generate_equipment(
        genre=TTRPGGenre.CYBERPUNK,
        npc_role=NPCRole.CRIMINAL,
        level=7,
        wealth_level="wealthy"
    )
    
    if "cyberware" in str(equipment.items):
        print("  ✓ Has cyberware")
    if "credits" in equipment.currency:
        print(f"  ✓ Has {equipment.currency['credits']} credits")
    
    # Test post-apocalyptic specific generation
    print("\nPost-Apocalyptic Scavenger:")
    print("-" * 40)
    
    name, _ = NameGenerator.generate_name(
        genre=TTRPGGenre.POST_APOCALYPTIC,
        include_nickname=True
    )
    print(f"  Name: {name}")
    
    equipment = EquipmentGenerator.generate_equipment(
        genre=TTRPGGenre.POST_APOCALYPTIC,
        level=5,
        wealth_level="poor"
    )
    
    if any("makeshift" in item.lower() or "scrap" in item.lower() for item in equipment.weapons):
        print("  ✓ Has makeshift weapons")
    if "caps" in equipment.currency:
        print(f"  ✓ Has {equipment.currency['caps']} caps")
    
    # Test western specific generation
    print("\nWestern Gunslinger:")
    print("-" * 40)
    
    name, _ = NameGenerator.generate_name(
        genre=TTRPGGenre.WESTERN,
        include_title=True,
        include_nickname=True
    )
    print(f"  Name: {name}")
    
    equipment = EquipmentGenerator.generate_equipment(
        genre=TTRPGGenre.WESTERN,
        level=8,
        wealth_level="standard"
    )
    
    if any("colt" in w.lower() or "winchester" in w.lower() for w in equipment.weapons):
        print("  ✓ Has period-appropriate weapons")
    if "dollars" in equipment.currency:
        print(f"  ✓ Has ${equipment.currency['dollars']}")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("GENRE-SPECIFIC GENERATION TEST SUITE")
    print("="*60)
    
    test_name_generation()
    test_equipment_generation()
    test_npc_generation()
    test_genre_specific_features()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()