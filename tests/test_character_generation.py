"""Tests for character generation functionality with genre support."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.character_generation.character_generator import CharacterGenerator
from src.character_generation.npc_generator import NPCGenerator
from src.character_generation.models import (
    Character,
    NPC,
    TTRPGGenre,
    CharacterClass,
    CharacterRace,
    NPCRole,
)
from src.character_generation.mcp_tools import (
    initialize_character_tools,
    register_character_tools,
)
from src.character_generation.validators import ValidationError


class TestCharacterGenerator:
    """Test the character generator with genre support."""

    @pytest.fixture
    def character_generator(self):
        """Create a character generator instance."""
        return CharacterGenerator()

    def test_generate_fantasy_character(self, character_generator):
        """Test generating a fantasy character."""
        character = character_generator.generate_character(
            system="D&D 5e",
            level=5,
            genre="fantasy"
        )
        
        assert character is not None
        assert character.system == "D&D 5e"
        assert character.stats.level == 5
        assert character.genre == TTRPGGenre.FANTASY
        assert character.name is not None
        assert character.character_class is not None
        assert character.race is not None

    def test_generate_cyberpunk_character(self, character_generator):
        """Test generating a cyberpunk character."""
        character = character_generator.generate_character(
            system="Cyberpunk 2020",
            level=3,
            genre="cyberpunk"
        )
        
        assert character.genre == TTRPGGenre.CYBERPUNK
        assert character.character_class in [
            CharacterClass.NETRUNNER, CharacterClass.SOLO, CharacterClass.FIXER,
            CharacterClass.CORPORATE, CharacterClass.ROCKERBOY, CharacterClass.TECHIE,
            CharacterClass.MEDIA, CharacterClass.COP, CharacterClass.NOMAD
        ]

    def test_generate_sci_fi_character(self, character_generator):
        """Test generating a sci-fi character."""
        character = character_generator.generate_character(
            system="Traveller",
            level=4,
            genre="sci_fi"
        )
        
        assert character.genre == TTRPGGenre.SCI_FI
        assert character.character_class in [
            CharacterClass.ENGINEER, CharacterClass.SCIENTIST, CharacterClass.PILOT,
            CharacterClass.MARINE, CharacterClass.DIPLOMAT, CharacterClass.XENOBIOLOGIST,
            CharacterClass.TECH_SPECIALIST, CharacterClass.PSION, CharacterClass.BOUNTY_HUNTER
        ]

    def test_genre_determination_from_system(self, character_generator):
        """Test automatic genre determination from system name."""
        # Fantasy systems
        character = character_generator.generate_character(system="Dungeons & Dragons 5e")
        assert character.genre == TTRPGGenre.FANTASY
        
        # Sci-Fi systems
        character = character_generator.generate_character(system="Star Wars RPG")
        assert character.genre == TTRPGGenre.SCI_FI
        
        # Cyberpunk systems
        character = character_generator.generate_character(system="Cyberpunk 2077")
        assert character.genre == TTRPGGenre.CYBERPUNK

    def test_extended_character_creation(self, character_generator):
        """Test creation of extended character with genre-specific data."""
        character = character_generator.generate_character(
            system="Shadowrun",
            level=3,
            genre="cyberpunk",
            use_extended=True
        )
        
        assert hasattr(character, 'genre_data')
        assert character.genre == TTRPGGenre.CYBERPUNK

    def test_stat_generation_methods(self, character_generator):
        """Test different stat generation methods."""
        # Standard array
        character = character_generator.generate_character(stat_generation="standard")
        assert character.stats is not None
        
        # Random stats
        character = character_generator.generate_character(stat_generation="random")
        assert character.stats is not None
        
        # Point buy
        character = character_generator.generate_character(stat_generation="point_buy")
        assert character.stats is not None

    def test_validation_error_handling(self, character_generator):
        """Test validation error handling."""
        with pytest.raises(ValidationError):
            character_generator.generate_character(level=-1)
        
        with pytest.raises(ValidationError):
            character_generator.generate_character(level=25)

    def test_name_generation_by_genre(self, character_generator):
        """Test genre-specific name generation."""
        # Fantasy names
        fantasy_char = character_generator.generate_character(genre="fantasy")
        assert fantasy_char.name is not None
        
        # Cyberpunk names
        cyber_char = character_generator.generate_character(genre="cyberpunk")
        assert cyber_char.name is not None
        
        # Western names
        western_char = character_generator.generate_character(genre="western")
        assert western_char.name is not None


class TestNPCGenerator:
    """Test the NPC generator with genre support."""

    @pytest.fixture
    def npc_generator(self):
        """Create an NPC generator instance."""
        return NPCGenerator()

    def test_generate_basic_npc(self, npc_generator):
        """Test generating a basic NPC."""
        npc = npc_generator.generate_npc(
            system="D&D 5e",
            role="merchant",
            importance="minor"
        )
        
        assert npc is not None
        assert npc.system == "D&D 5e"
        assert npc.role == NPCRole.MERCHANT
        assert npc.importance == "Minor"
        assert npc.name is not None

    def test_generate_npc_with_genre(self, npc_generator):
        """Test generating NPCs with genre specification."""
        npc = npc_generator.generate_npc(
            system="Cyberpunk 2020",
            role="corporate",
            genre="cyberpunk"
        )
        
        assert npc.genre == TTRPGGenre.CYBERPUNK
        assert npc.name is not None

    def test_npc_level_calculation(self, npc_generator):
        """Test NPC level calculation based on party level."""
        # Minor NPC should be lower level
        minor_npc = npc_generator.generate_npc(
            role="commoner",
            importance="minor",
            party_level=5
        )
        
        # Major NPC should be higher level
        major_npc = npc_generator.generate_npc(
            role="noble",
            importance="major",
            party_level=5
        )
        
        assert major_npc.stats.level >= minor_npc.stats.level

    def test_genre_specific_roles(self, npc_generator):
        """Test genre-appropriate role selection."""
        # Test cyberpunk roles
        cyber_npc = npc_generator.generate_npc(
            system="Cyberpunk 2020",
            genre="cyberpunk"
        )
        assert cyber_npc.genre == TTRPGGenre.CYBERPUNK
        
        # Test western roles
        western_npc = npc_generator.generate_npc(
            system="Deadlands",
            genre="western"
        )
        assert western_npc.genre == TTRPGGenre.WESTERN

    def test_npc_equipment_by_genre(self, npc_generator):
        """Test genre-specific equipment generation."""
        # Fantasy NPC
        fantasy_npc = npc_generator.generate_npc(
            role="guard",
            genre="fantasy"
        )
        assert fantasy_npc.equipment is not None
        
        # Cyberpunk NPC
        cyber_npc = npc_generator.generate_npc(
            role="corporate",
            genre="cyberpunk"
        )
        assert cyber_npc.equipment is not None

    def test_npc_personality_traits(self, npc_generator):
        """Test NPC personality trait generation."""
        npc = npc_generator.generate_npc(
            role="merchant",
            importance="supporting"
        )
        
        assert npc.personality_traits is not None
        assert len(npc.personality_traits) > 0

    def test_npc_backstory_generation(self, npc_generator):
        """Test NPC backstory generation."""
        npc = npc_generator.generate_npc(
            role="scholar",
            importance="major",
            backstory_depth="detailed"
        )
        
        assert npc.backstory is not None

    def test_npc_secrets_for_important_npcs(self, npc_generator):
        """Test that important NPCs get secrets."""
        supporting_npc = npc_generator.generate_npc(
            role="noble",
            importance="supporting"
        )
        
        major_npc = npc_generator.generate_npc(
            role="criminal",
            importance="major"
        )
        
        assert supporting_npc.secrets is not None
        assert major_npc.secrets is not None
        assert len(major_npc.secrets) >= len(supporting_npc.secrets)

    def test_genre_name_generation(self, npc_generator):
        """Test genre-specific name generation for NPCs."""
        # Test different genres generate appropriate names
        genres_to_test = [
            ("fantasy", TTRPGGenre.FANTASY),
            ("cyberpunk", TTRPGGenre.CYBERPUNK),
            ("sci_fi", TTRPGGenre.SCI_FI),
            ("western", TTRPGGenre.WESTERN),
            ("cosmic_horror", TTRPGGenre.COSMIC_HORROR),
        ]
        
        for genre_str, genre_enum in genres_to_test:
            npc = npc_generator.generate_npc(
                role="merchant",
                genre=genre_str
            )
            assert npc.genre == genre_enum
            assert npc.name is not None
            assert len(npc.name) > 0


class TestMCPTools:
    """Test the MCP tools for character generation."""

    @pytest.fixture
    def mock_mcp_server(self):
        """Create a mock MCP server."""
        server = Mock()
        server.tool = Mock(return_value=lambda func: func)  # Decorator that returns the function
        return server

    @pytest.fixture
    def mock_db(self):
        """Create a mock database."""
        db = AsyncMock()
        db.add_document = AsyncMock()
        db.query_collection = AsyncMock(return_value={"documents": []})
        return db

    def test_initialize_character_tools(self, mock_db):
        """Test initialization of character tools."""
        # Should not raise an exception
        initialize_character_tools(mock_db)

    def test_register_character_tools(self, mock_mcp_server, mock_db):
        """Test registration of character tools."""
        initialize_character_tools(mock_db)
        
        # Should not raise an exception
        register_character_tools(mock_mcp_server)
        
        # Verify tools were registered
        assert mock_mcp_server.tool.called

    @pytest.mark.asyncio
    async def test_generate_character_tool(self, mock_db):
        """Test the generate_character MCP tool."""
        initialize_character_tools(mock_db)
        
        # Import the tool function
        from src.character_generation.mcp_tools import register_character_tools
        
        # Mock the MCP server to capture the registered function
        tools = {}
        
        def mock_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func
            return decorator
        
        mock_server = Mock()
        mock_server.tool = mock_tool
        
        register_character_tools(mock_server)
        
        # Test the generate_character tool
        generate_character = tools.get('generate_character')
        assert generate_character is not None
        
        result = await generate_character(
            system="D&D 5e",
            level=3,
            genre="fantasy"
        )
        
        assert result["success"] is True
        assert "character" in result

    @pytest.mark.asyncio
    async def test_generate_npc_tool(self, mock_db):
        """Test the generate_npc MCP tool."""
        initialize_character_tools(mock_db)
        
        # Mock the MCP server to capture the registered function
        tools = {}
        
        def mock_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func
            return decorator
        
        mock_server = Mock()
        mock_server.tool = mock_tool
        
        register_character_tools(mock_server)
        
        # Test the generate_npc tool
        generate_npc = tools.get('generate_npc')
        assert generate_npc is not None
        
        result = await generate_npc(
            system="D&D 5e",
            role="merchant",
            genre="fantasy"
        )
        
        assert result["success"] is True
        assert "npc" in result

    @pytest.mark.asyncio
    async def test_list_available_genres_tool(self, mock_db):
        """Test the list_available_genres MCP tool."""
        initialize_character_tools(mock_db)
        
        # Mock the MCP server to capture the registered function
        tools = {}
        
        def mock_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func
            return decorator
        
        mock_server = Mock()
        mock_server.tool = mock_tool
        
        register_character_tools(mock_server)
        
        # Test the list_available_genres tool
        list_genres = tools.get('list_available_genres')
        assert list_genres is not None
        
        result = await list_genres()
        
        assert result["success"] is True
        assert "genres" in result
        assert len(result["genres"]) > 0

    @pytest.mark.asyncio
    async def test_get_genre_content_tool(self, mock_db):
        """Test the get_genre_content MCP tool."""
        initialize_character_tools(mock_db)
        
        # Mock the MCP server to capture the registered function
        tools = {}
        
        def mock_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func
            return decorator
        
        mock_server = Mock()
        mock_server.tool = mock_tool
        
        register_character_tools(mock_server)
        
        # Test the get_genre_content tool
        get_content = tools.get('get_genre_content')
        assert get_content is not None
        
        result = await get_content(genre="cyberpunk")
        
        assert result["success"] is True
        assert "content" in result

    @pytest.mark.asyncio
    async def test_validation_error_handling(self, mock_db):
        """Test validation error handling in MCP tools."""
        initialize_character_tools(mock_db)
        
        tools = {}
        
        def mock_tool():
            def decorator(func):
                tools[func.__name__] = func
                return func
            return decorator
        
        mock_server = Mock()
        mock_server.tool = mock_tool
        
        register_character_tools(mock_server)
        
        # Test with invalid level
        generate_character = tools.get('generate_character')
        result = await generate_character(level=-1)
        
        assert result["success"] is False
        assert "error" in result


class TestGenreIntegration:
    """Test integration of genre support across all components."""

    @pytest.fixture
    def character_generator(self):
        """Create a character generator instance."""
        return CharacterGenerator()

    @pytest.fixture
    def npc_generator(self):
        """Create an NPC generator instance."""
        return NPCGenerator()

    def test_all_genres_supported(self, character_generator, npc_generator):
        """Test that all TTRPGGenre enum values are supported."""
        supported_genres = [
            TTRPGGenre.FANTASY,
            TTRPGGenre.SCI_FI,
            TTRPGGenre.CYBERPUNK,
            TTRPGGenre.COSMIC_HORROR,
            TTRPGGenre.POST_APOCALYPTIC,
            TTRPGGenre.SUPERHERO,
            TTRPGGenre.WESTERN,
            TTRPGGenre.STEAMPUNK,
            TTRPGGenre.MODERN,
        ]
        
        for genre in supported_genres:
            # Test character generation
            character = character_generator.generate_character(
                genre=genre.value
            )
            assert character.genre == genre
            
            # Test NPC generation
            npc = npc_generator.generate_npc(
                genre=genre.value
            )
            assert npc.genre == genre

    def test_genre_consistency(self, character_generator, npc_generator):
        """Test that genre determination is consistent."""
        test_systems = [
            ("D&D 5e", TTRPGGenre.FANTASY),
            ("Cyberpunk 2020", TTRPGGenre.CYBERPUNK),
            ("Traveller", TTRPGGenre.SCI_FI),
            ("Call of Cthulhu", TTRPGGenre.COSMIC_HORROR),
            ("Deadlands", TTRPGGenre.WESTERN),
        ]
        
        for system, expected_genre in test_systems:
            character = character_generator.generate_character(system=system)
            npc = npc_generator.generate_npc(system=system)
            
            assert character.genre == expected_genre
            assert npc.genre == expected_genre

    def test_equipment_varies_by_genre(self, npc_generator):
        """Test that equipment varies appropriately by genre."""
        fantasy_npc = npc_generator.generate_npc(
            role="guard",
            genre="fantasy"
        )
        
        cyberpunk_npc = npc_generator.generate_npc(
            role="corporate",
            genre="cyberpunk"
        )
        
        # Equipment should be different for different genres
        fantasy_equipment = fantasy_npc.equipment.weapons + fantasy_npc.equipment.items
        cyberpunk_equipment = cyberpunk_npc.equipment.weapons + cyberpunk_npc.equipment.items
        
        # There should be some differences in equipment
        assert fantasy_equipment != cyberpunk_equipment

    def test_names_vary_by_genre(self, npc_generator):
        """Test that names vary appropriately by genre."""
        # Generate multiple NPCs of the same role but different genres
        fantasy_names = set()
        cyberpunk_names = set()
        
        for _ in range(10):
            fantasy_npc = npc_generator.generate_npc(
                role="merchant",
                genre="fantasy"
            )
            fantasy_names.add(fantasy_npc.name)
            
            cyberpunk_npc = npc_generator.generate_npc(
                role="merchant",
                genre="cyberpunk"
            )
            cyberpunk_names.add(cyberpunk_npc.name)
        
        # Names should be different between genres
        assert len(fantasy_names.intersection(cyberpunk_names)) == 0


# Performance and edge case tests
class TestPerformanceAndEdgeCases:
    """Test performance and edge cases."""

    @pytest.fixture
    def character_generator(self):
        """Create a character generator instance."""
        return CharacterGenerator()

    def test_rapid_character_generation(self, character_generator):
        """Test rapid generation of multiple characters."""
        characters = []
        for _ in range(50):
            character = character_generator.generate_character()
            characters.append(character)
        
        assert len(characters) == 50
        # All characters should be unique
        names = [c.name for c in characters]
        assert len(set(names)) > len(names) * 0.8  # At least 80% unique

    def test_invalid_genre_handling(self, character_generator):
        """Test handling of invalid genre values."""
        # Invalid genre should default to fantasy
        character = character_generator.generate_character(genre="invalid_genre")
        assert character.genre == TTRPGGenre.FANTASY

    def test_missing_optional_parameters(self, character_generator):
        """Test generation with minimal parameters."""
        character = character_generator.generate_character()
        
        assert character is not None
        assert character.name is not None
        assert character.character_class is not None
        assert character.race is not None
        assert character.stats is not None

    @pytest.mark.slow
    def test_memory_usage_stability(self, character_generator):
        """Test that memory usage remains stable during generation."""
        import gc
        import sys
        
        # Force garbage collection
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Generate many characters
        for _ in range(100):
            character = character_generator.generate_character()
            del character
        
        # Force garbage collection again
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Object count should not grow significantly
        growth_rate = (final_objects - initial_objects) / initial_objects
        assert growth_rate < 0.1  # Less than 10% growth


if __name__ == "__main__":
    pytest.main([__file__])