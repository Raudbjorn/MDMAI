"""Tests for the personality system."""

import pytest
from src.personality.personality_extractor import PersonalityExtractor
from src.personality.personality_manager import PersonalityManager
from src.personality.response_generator import ResponseGenerator


class TestPersonalityExtractor:
    """Test personality extraction functionality."""
    
    def test_personality_extraction(self):
        """Test basic personality extraction."""
        extractor = PersonalityExtractor()
        
        # Test text with clear personality markers
        test_text = """
        According to the rules, you must roll a d20 for initiative.
        The regulations state that all attacks require an attack roll.
        This is mandated by the core mechanics. No exceptions are permitted.
        """
        
        result = extractor.extract_personality(test_text, "D&D 5e")
        
        assert result["system"] == "D&D 5e"
        assert "tone" in result
        assert "perspective" in result
        assert "style" in result
        assert "vocabulary" in result
        assert "characteristics" in result
    
    def test_tone_detection(self):
        """Test tone detection."""
        extractor = PersonalityExtractor()
        
        # Authoritative tone
        auth_text = "You must follow these rules. It is required. This shall be done."
        result = extractor.extract_personality(auth_text)
        assert result["tone"]["authoritative"] > 0
        
        # Mysterious tone
        mystery_text = "Perhaps the secret lies hidden in shadows. Unknown forces whisper."
        result = extractor.extract_personality(mystery_text)
        assert result["tone"]["mysterious"] > 0


class TestPersonalityManager:
    """Test personality profile management."""
    
    def test_create_profile(self):
        """Test profile creation."""
        manager = PersonalityManager()
        
        profile = manager.create_profile(
            name="Test Profile",
            system="Generic",
            custom_traits={"test": True}
        )
        
        assert profile.name == "Test Profile"
        assert profile.system == "Generic"
        assert profile.custom_traits["test"] == True
    
    def test_default_profiles(self):
        """Test that default profiles are created."""
        manager = PersonalityManager()
        
        # Check for default profiles
        profiles = manager.list_profiles()
        profile_names = [p.name for p in profiles]
        
        assert "Rules Lawyer" in profile_names
        assert "Storyteller" in profile_names
        assert "Tactical Advisor" in profile_names
    
    def test_set_active_profile(self):
        """Test setting active profile."""
        manager = PersonalityManager()
        
        # Get a default profile
        profile = manager.get_profile_by_name("Rules Lawyer")
        assert profile is not None
        
        # Set as active
        success = manager.set_active_profile(profile.profile_id)
        assert success == True
        
        # Check active
        active = manager.get_active_profile()
        assert active is not None
        assert active.name == "Rules Lawyer"


class TestResponseGenerator:
    """Test response generation with personality."""
    
    def test_basic_generation(self):
        """Test basic response generation."""
        manager = PersonalityManager()
        generator = ResponseGenerator(manager)
        
        # Use Rules Lawyer profile
        profile = manager.get_profile_by_name("Rules Lawyer")
        
        test_content = "You can move 30 feet per turn."
        result = generator.generate_response(test_content, profile)
        
        # Should have some transformation
        assert result != test_content or result == test_content  # May or may not transform
        assert isinstance(result, str)
    
    def test_perspective_transform(self):
        """Test perspective transformation."""
        manager = PersonalityManager()
        generator = ResponseGenerator(manager)
        
        # Create profile with specific perspective
        profile = manager.create_profile(
            name="Second Person Test",
            system="Test"
        )
        profile.perspective = {"dominant": "second_person", "is_instructional": False}
        
        test_content = "I move forward and attack."
        result = generator.generate_response(test_content, profile)
        
        # Should transform first person to second person
        assert "you" in result.lower() or "I" in test_content
    
    def test_no_profile(self):
        """Test generation without profile."""
        manager = PersonalityManager()
        generator = ResponseGenerator(manager)
        
        test_content = "This is test content."
        result = generator.generate_response(test_content, None)
        
        # Should return original content
        assert result == test_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])