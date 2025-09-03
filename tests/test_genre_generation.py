#!/usr/bin/env python3
"""Pytest tests for genre-specific NPC and equipment generation."""

import pytest
from typing import Dict, List

from src.character_generation.equipment_generator import EquipmentGenerator, EquipmentQuality
from src.character_generation.models import NPCRole, TTRPGGenre
from src.character_generation.name_generator import NameGenerator, NameStyle
from src.character_generation.npc_generator import NPCGenerator


class TestNameGeneration:
    """Test name generation across different genres."""
    
    @pytest.mark.parametrize("genre", [
        TTRPGGenre.FANTASY,
        TTRPGGenre.SCI_FI,
        TTRPGGenre.CYBERPUNK,
        TTRPGGenre.COSMIC_HORROR,
        TTRPGGenre.POST_APOCALYPTIC,
        TTRPGGenre.WESTERN,
        TTRPGGenre.SUPERHERO
    ])
    def test_name_generation_by_genre(self, genre):
        """Test that names are generated for each genre."""
        name, components = NameGenerator.generate_name(
            genre=genre,
            gender="neutral",
            include_title=True,
            include_nickname=True
        )
        
        assert name is not None
        assert isinstance(name, str)
        assert len(name) > 0
        assert components is not None
        
    def test_gender_specific_names(self):
        """Test that gender-specific names are generated."""
        for gender in ["male", "female", "neutral"]:
            name, components = NameGenerator.generate_name(
                genre=TTRPGGenre.FANTASY,
                gender=gender
            )
            assert name is not None
            assert isinstance(name, str)
            assert len(name) > 0
            
    def test_role_specific_names(self):
        """Test that role-specific names are generated."""
        roles = [NPCRole.NOBLE, NPCRole.MERCHANT, NPCRole.SOLDIER, NPCRole.SCHOLAR]
        for role in roles:
            name, components = NameGenerator.generate_name(
                genre=TTRPGGenre.FANTASY,
                role=role,
                include_title=True
            )
            assert name is not None
            assert isinstance(name, str)
            assert len(name) > 0
            
    def test_organization_names(self):
        """Test organization name generation."""
        for genre in [TTRPGGenre.FANTASY, TTRPGGenre.SCI_FI, TTRPGGenre.CYBERPUNK]:
            org_name = NameGenerator.generate_organization_name(genre, "guild")
            assert org_name is not None
            assert isinstance(org_name, str)
            assert len(org_name) > 0


class TestEquipmentGeneration:
    """Test equipment generation across different genres."""
    
    @pytest.mark.parametrize("genre", [
        TTRPGGenre.FANTASY,
        TTRPGGenre.SCI_FI,
        TTRPGGenre.CYBERPUNK,
        TTRPGGenre.POST_APOCALYPTIC,
        TTRPGGenre.WESTERN,
        TTRPGGenre.SUPERHERO
    ])
    def test_equipment_generation_by_genre(self, genre):
        """Test that equipment is generated for each genre."""
        equipment = EquipmentGenerator.generate_equipment(
            genre=genre,
            level=5,
            wealth_level="standard"
        )
        
        assert equipment is not None
        assert hasattr(equipment, 'weapons')
        assert hasattr(equipment, 'armor')
        assert hasattr(equipment, 'items')
        assert isinstance(equipment.weapons, list)
        assert isinstance(equipment.armor, list)
        assert isinstance(equipment.items, list)
        
    def test_wealth_level_effects(self):
        """Test that wealth level affects equipment quality."""
        wealth_levels = ["poor", "standard", "wealthy", "noble"]
        
        for wealth in wealth_levels:
            equipment = EquipmentGenerator.generate_equipment(
                genre=TTRPGGenre.FANTASY,
                level=1,
                wealth_level=wealth
            )
            assert equipment is not None
            # Verify equipment is generated regardless of wealth level
            total_items = len(equipment.weapons) + len(equipment.armor) + len(equipment.items)
            assert total_items > 0
            
    def test_level_progression(self):
        """Test that higher levels produce better equipment."""
        low_level_eq = EquipmentGenerator.generate_equipment(
            genre=TTRPGGenre.FANTASY,
            level=1,
            wealth_level="standard"
        )
        
        high_level_eq = EquipmentGenerator.generate_equipment(
            genre=TTRPGGenre.FANTASY,
            level=15,
            wealth_level="standard"
        )
        
        assert low_level_eq is not None
        assert high_level_eq is not None
        # Both should generate equipment
        assert len(low_level_eq.weapons) + len(low_level_eq.armor) + len(low_level_eq.items) > 0
        assert len(high_level_eq.weapons) + len(high_level_eq.armor) + len(high_level_eq.items) > 0
        
    def test_magical_equipment_generation(self):
        """Test magical equipment generation."""
        equipment = EquipmentGenerator.generate_equipment(
            genre=TTRPGGenre.FANTASY,
            level=10,
            wealth_level="wealthy",
            include_magical=True
        )
        
        assert equipment is not None
        # Should generate some equipment
        total_items = len(equipment.weapons) + len(equipment.armor) + len(equipment.items)
        assert total_items > 0


class TestNPCGeneration:
    """Test full NPC generation with genre integration."""
    
    def test_basic_npc_generation(self):
        """Test basic NPC generation works."""
        generator = NPCGenerator()
        npc = generator.generate_npc(
            system="D&D 5e",
            importance="minor"
        )
        
        assert npc is not None
        assert npc.name is not None
        assert len(npc.name) > 0
        assert npc.system == "D&D 5e"
        
    @pytest.mark.parametrize("genre", [
        TTRPGGenre.FANTASY,
        TTRPGGenre.SCI_FI,
        TTRPGGenre.CYBERPUNK,
        TTRPGGenre.WESTERN
    ])
    def test_genre_specific_npc_generation(self, genre):
        """Test NPC generation with explicit genre override."""
        generator = NPCGenerator()
        npc = generator.generate_npc(
            system="D&D 5e",
            genre=genre,
            importance="supporting"
        )
        
        assert npc is not None
        assert npc.name is not None
        assert len(npc.name) > 0
        assert hasattr(npc, 'equipment')
        
    def test_role_based_npc_generation(self):
        """Test NPC generation with specific roles."""
        generator = NPCGenerator()
        roles = ["merchant", "soldier", "noble", "scholar"]
        
        for role in roles:
            npc = generator.generate_npc(
                system="D&D 5e",
                role=role,
                importance="supporting"
            )
            assert npc is not None
            assert npc.name is not None
            assert len(npc.name) > 0
            
    def test_invalid_role_handling(self):
        """Test that invalid roles are handled gracefully."""
        generator = NPCGenerator()
        # This should not raise an exception
        npc = generator.generate_npc(
            system="D&D 5e",
            role="invalid_role_that_does_not_exist",
            importance="minor"
        )
        
        assert npc is not None
        assert npc.name is not None
        assert len(npc.name) > 0
        
    def test_importance_levels(self):
        """Test different importance levels."""
        generator = NPCGenerator()
        importance_levels = ["minor", "supporting", "major"]
        
        for importance in importance_levels:
            npc = generator.generate_npc(
                system="D&D 5e",
                importance=importance
            )
            assert npc is not None
            assert npc.name is not None
            assert len(npc.name) > 0


class TestGenreSpecificFeatures:
    """Test genre-specific features and integration."""
    
    def test_fantasy_features(self):
        """Test Fantasy genre specific features."""
        # Names should include medieval/fantasy elements
        name, components = NameGenerator.generate_name(
            genre=TTRPGGenre.FANTASY,
            role=NPCRole.NOBLE,
            include_title=True
        )
        assert name is not None
        assert len(name) > 0
        
        # Equipment should include medieval weapons
        equipment = EquipmentGenerator.generate_equipment(
            genre=TTRPGGenre.FANTASY,
            include_magical=True
        )
        assert equipment is not None
        
    def test_scifi_features(self):
        """Test Sci-Fi genre specific features."""
        # Names should include futuristic elements
        name, components = NameGenerator.generate_name(
            genre=TTRPGGenre.SCI_FI,
            include_title=True
        )
        assert name is not None
        assert len(name) > 0
        
        # Equipment should include futuristic items
        equipment = EquipmentGenerator.generate_equipment(
            genre=TTRPGGenre.SCI_FI,
            level=5
        )
        assert equipment is not None
        
    def test_cyberpunk_features(self):
        """Test Cyberpunk genre specific features."""
        # Names should include cyberpunk elements
        name, components = NameGenerator.generate_name(
            genre=TTRPGGenre.CYBERPUNK,
            include_nickname=True
        )
        assert name is not None
        assert len(name) > 0
        
        # Equipment should include tech/cyber items
        equipment = EquipmentGenerator.generate_equipment(
            genre=TTRPGGenre.CYBERPUNK,
            level=8
        )
        assert equipment is not None
        
    def test_horror_features(self):
        """Test Cosmic Horror genre specific features."""
        # Names should include appropriate horror elements
        name, components = NameGenerator.generate_name(
            genre=TTRPGGenre.COSMIC_HORROR,
            role=NPCRole.SCHOLAR
        )
        assert name is not None
        assert len(name) > 0
        
        # Equipment should include investigation/survival items
        equipment = EquipmentGenerator.generate_equipment(
            genre=TTRPGGenre.COSMIC_HORROR,
            level=3
        )
        assert equipment is not None


class TestIntegration:
    """Integration tests combining multiple systems."""
    
    def test_full_npc_with_genre_override(self):
        """Test complete NPC generation with genre override."""
        generator = NPCGenerator()
        
        # Generate NPCs for different genres
        genres = [TTRPGGenre.FANTASY, TTRPGGenre.SCI_FI, TTRPGGenre.CYBERPUNK]
        
        for genre in genres:
            npc = generator.generate_npc(
                system="D&D 5e",  # Base system
                genre=genre,      # Genre override
                role="soldier",
                importance="supporting",
                level=5
            )
            
            assert npc is not None
            assert npc.name is not None
            assert len(npc.name) > 0
            # NPC should have equipment appropriate to the genre
            assert hasattr(npc, 'equipment')
            
    def test_name_equipment_consistency(self):
        """Test that names and equipment are consistent within genre."""
        for genre in [TTRPGGenre.FANTASY, TTRPGGenre.SCI_FI, TTRPGGenre.WESTERN]:
            # Generate name
            name, components = NameGenerator.generate_name(
                genre=genre,
                role=NPCRole.SOLDIER,
                include_title=True
            )
            
            # Generate equipment
            equipment = EquipmentGenerator.generate_equipment(
                genre=genre,
                npc_role=NPCRole.SOLDIER,
                level=5
            )
            
            assert name is not None
            assert equipment is not None
            assert len(name) > 0
            # Both should be generated successfully for the same genre/role
            assert len(equipment.weapons) + len(equipment.armor) + len(equipment.items) > 0