"""Voice profile mapper for converting personality and NPC traits to voice profiles."""

from typing import Any, Dict, List, Optional

from structlog import get_logger

from .models import (
    VoiceAge,
    VoiceEmotion,
    VoiceGender,
    VoiceProfile,
)

logger = get_logger(__name__)


# Tone to default emotion mapping
TONE_EMOTION_MAP: Dict[str, VoiceEmotion] = {
    "authoritative": VoiceEmotion.AUTHORITATIVE,
    "mysterious": VoiceEmotion.MYSTERIOUS,
    "whimsical": VoiceEmotion.WHIMSICAL,
    "ominous": VoiceEmotion.OMINOUS,
    "formal": VoiceEmotion.NEUTRAL,
    "casual": VoiceEmotion.HAPPY,
    "military": VoiceEmotion.AUTHORITATIVE,
    "scholarly": VoiceEmotion.NEUTRAL,
    "dramatic": VoiceEmotion.EXCITED,
    "sinister": VoiceEmotion.OMINOUS,
    "friendly": VoiceEmotion.HAPPY,
    "solemn": VoiceEmotion.CALM,
}

# Personality characteristics to voice parameter adjustments
CHARACTERISTIC_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
    # Voice lowering/deepening
    "commanding": {"pitch": -0.15, "energy": 0.2},
    "gruff": {"pitch": -0.2, "roughness": 0.3},
    "stern": {"pitch": -0.1, "pitch_variance": -0.2},
    "deep": {"pitch": -0.25},
    "booming": {"pitch": -0.2, "energy": 0.3},

    # Voice raising/brightening
    "bright": {"pitch": 0.15, "energy": 0.1},
    "cheerful": {"pitch": 0.1, "pitch_variance": 0.15},
    "youthful": {"pitch": 0.1, "speed": 0.05},
    "high-pitched": {"pitch": 0.2},

    # Energy/expressiveness
    "enigmatic": {"pitch_variance": -0.2, "breathiness": 0.15},
    "academic": {"speed": -0.1, "pitch_variance": -0.1},
    "formal": {"speed": -0.05, "energy": -0.1},
    "casual": {"speed": 0.1, "pitch_variance": 0.1},
    "optimistic": {"pitch": 0.1, "energy": 0.15},
    "pessimistic": {"pitch": -0.1, "energy": -0.15},
    "concise": {"speed": 0.15},
    "elaborate": {"speed": -0.1},
    "dramatic": {"pitch_variance": 0.2, "energy": 0.2},
    "monotone": {"pitch_variance": -0.3},

    # Voice quality
    "raspy": {"roughness": 0.25},
    "smooth": {"roughness": -0.1, "warmth": 0.15},
    "breathy": {"breathiness": 0.2},
    "warm": {"warmth": 0.2},
    "cold": {"warmth": -0.2},
    "soft": {"energy": -0.2, "breathiness": 0.1},
    "loud": {"energy": 0.25},

    # Speaking pace
    "slow": {"speed": -0.15},
    "fast": {"speed": 0.15},
    "measured": {"speed": -0.1, "pitch_variance": -0.1},
    "hurried": {"speed": 0.2, "pitch_variance": 0.1},
}

# NPC role to voice characteristics
NPC_ROLE_VOICE_MAP: Dict[str, Dict[str, Any]] = {
    # Fantasy roles
    "merchant": {"speed": 0.6, "energy": 0.6, "pitch_variance": 0.4, "warmth": 0.6},
    "guard": {"pitch": 0.3, "energy": 0.6, "roughness": 0.2},
    "noble": {"speed": 0.4, "pitch": 0.6, "pitch_variance": 0.2, "warmth": 0.4},
    "scholar": {"speed": 0.4, "pitch_variance": 0.2},
    "criminal": {"pitch": 0.4, "roughness": 0.3, "breathiness": 0.1},
    "innkeeper": {"energy": 0.7, "pitch_variance": 0.5, "warmth": 0.7},
    "priest": {"speed": 0.4, "pitch_variance": 0.3, "warmth": 0.5},
    "adventurer": {"energy": 0.6, "pitch_variance": 0.4},
    "mage": {"speed": 0.4, "breathiness": 0.2, "pitch": 0.55},
    "assassin": {"speed": 0.5, "pitch": 0.4, "breathiness": 0.2, "energy": 0.3},
    "blacksmith": {"pitch": 0.3, "roughness": 0.25, "energy": 0.6},
    "bard": {"pitch_variance": 0.5, "energy": 0.6, "warmth": 0.6},
    "beggar": {"pitch": 0.4, "energy": 0.3, "roughness": 0.2},
    "knight": {"pitch": 0.35, "energy": 0.55, "speed": 0.45},
    "healer": {"warmth": 0.7, "speed": 0.45, "energy": 0.4},

    # Sci-fi roles
    "scientist": {"speed": 0.45, "pitch_variance": 0.2},
    "soldier": {"pitch": 0.35, "energy": 0.6, "speed": 0.55},
    "pilot": {"speed": 0.55, "energy": 0.55},
    "android": {"pitch_variance": 0.1, "speed": 0.5, "warmth": 0.2},
    "alien": {"pitch_variance": 0.4, "breathiness": 0.3},

    # Cyberpunk roles
    "fixer": {"speed": 0.55, "energy": 0.5, "roughness": 0.15},
    "netrunner": {"speed": 0.6, "pitch_variance": 0.3},
    "corpo": {"speed": 0.45, "energy": 0.4, "warmth": 0.3},
    "street_vendor": {"speed": 0.65, "energy": 0.7, "pitch_variance": 0.4},

    # Horror roles
    "cultist": {"breathiness": 0.25, "pitch_variance": 0.3, "warmth": 0.2},
    "investigator": {"speed": 0.45, "pitch_variance": 0.2},
    "asylum_patient": {"pitch_variance": 0.4, "energy": 0.3, "breathiness": 0.2},
}


class VoiceProfileMapper:
    """Maps personality profiles and NPC traits to voice profiles.

    This class provides static methods to create VoiceProfiles from:
    - PersonalityProfile objects (from the personality system)
    - NPC model objects (from the campaign system)
    - Raw trait dictionaries

    The mapping uses heuristics based on personality characteristics,
    tone, style, and NPC role to determine appropriate voice parameters.
    """

    @classmethod
    def from_personality_profile(
        cls,
        personality: Any,  # PersonalityProfile from personality_manager
        name: Optional[str] = None,
    ) -> VoiceProfile:
        """Create VoiceProfile from a PersonalityProfile.

        Args:
            personality: PersonalityProfile object
            name: Override name for the voice profile

        Returns:
            VoiceProfile with mapped parameters
        """
        # Determine default emotion from dominant tone
        tone_dict = getattr(personality, "tone", {})
        default_emotion = VoiceEmotion.NEUTRAL

        # Find dominant tone
        if tone_dict:
            dominant_tone = max(tone_dict.items(), key=lambda x: x[1], default=("neutral", 0))
            tone_name = dominant_tone[0].lower()
            default_emotion = TONE_EMOTION_MAP.get(tone_name, VoiceEmotion.NEUTRAL)

        # Start with baseline profile
        profile = VoiceProfile(
            name=name or getattr(personality, "name", "Generated"),
            default_emotion=default_emotion,
            personality_profile_id=getattr(personality, "profile_id", None),
        )

        # Apply characteristic adjustments
        characteristics = getattr(personality, "characteristics", [])
        for characteristic in characteristics:
            char_lower = characteristic.lower()
            if char_lower in CHARACTERISTIC_ADJUSTMENTS:
                adjustments = CHARACTERISTIC_ADJUSTMENTS[char_lower]
                cls._apply_adjustments(profile, adjustments)

        # Adjust based on formality (from style)
        style = getattr(personality, "style", {})
        formality = style.get("formality", 0.5)
        if formality > 0.7:
            profile.speed = max(0.0, profile.speed - 0.1)
            profile.pitch_variance = max(0.0, profile.pitch_variance - 0.1)
        elif formality < 0.3:
            profile.speed = min(1.0, profile.speed + 0.05)
            profile.pitch_variance = min(1.0, profile.pitch_variance + 0.1)

        # Adjust based on sentiment
        sentiment = getattr(personality, "sentiment", {})
        polarity = sentiment.get("polarity", 0)
        if polarity > 0.3:
            profile.warmth = min(1.0, profile.warmth + 0.15)
            profile.energy = min(1.0, profile.energy + 0.1)
        elif polarity < -0.3:
            profile.warmth = max(0.0, profile.warmth - 0.15)

        return profile

    @classmethod
    def from_npc(
        cls,
        npc: Any,  # NPC model from campaign/models.py or character_generation/models.py
        campaign_id: Optional[str] = None,
    ) -> VoiceProfile:
        """Create VoiceProfile from an NPC model.

        Args:
            npc: NPC object with personality_traits, role, etc.
            campaign_id: Optional campaign context

        Returns:
            VoiceProfile with mapped parameters
        """
        profile = VoiceProfile(
            name=f"{getattr(npc, 'name', 'NPC')}_voice",
            npc_id=getattr(npc, "id", None),
            campaign_id=campaign_id,
        )

        # Get NPC role
        role = getattr(npc, "role", None)
        if role:
            # Handle enum or string
            role_str = role.value if hasattr(role, "value") else str(role).lower()

            if role_str in NPC_ROLE_VOICE_MAP:
                cls._apply_adjustments(profile, NPC_ROLE_VOICE_MAP[role_str])

        # Apply personality traits
        personality_traits = getattr(npc, "personality_traits", [])
        for trait in personality_traits:
            # Handle trait objects or strings
            if hasattr(trait, "trait"):
                trait_value = trait.trait.lower()
            else:
                trait_value = str(trait).lower()

            # Check for trait-based adjustments
            for keyword, adjustments in CHARACTERISTIC_ADJUSTMENTS.items():
                if keyword in trait_value:
                    cls._apply_adjustments(profile, adjustments)
                    break

            # Specific trait mappings
            if "gruff" in trait_value or "stern" in trait_value:
                profile.roughness = min(1.0, profile.roughness + 0.2)
                profile.pitch = max(0.0, profile.pitch - 0.1)
            elif "friendly" in trait_value or "welcoming" in trait_value:
                profile.energy = min(1.0, profile.energy + 0.1)
                profile.warmth = min(1.0, profile.warmth + 0.15)
                profile.pitch_variance = min(1.0, profile.pitch_variance + 0.1)
            elif "suspicious" in trait_value or "paranoid" in trait_value:
                profile.speed = max(0.0, profile.speed - 0.1)
                profile.pitch_variance = max(0.0, profile.pitch_variance - 0.1)
            elif "jovial" in trait_value or "merry" in trait_value:
                profile.energy = min(1.0, profile.energy + 0.2)
                profile.default_emotion = VoiceEmotion.HAPPY
            elif "menacing" in trait_value or "threatening" in trait_value:
                profile.pitch = max(0.0, profile.pitch - 0.15)
                profile.default_emotion = VoiceEmotion.OMINOUS

        # Infer gender from NPC data if available
        description = getattr(npc, "description", "") or ""
        gender_hint = cls._infer_gender(description, npc)
        if gender_hint:
            profile.gender = gender_hint

        return profile

    @classmethod
    def from_traits(
        cls,
        traits: Dict[str, Any],
        name: str = "Custom",
    ) -> VoiceProfile:
        """Create VoiceProfile from a traits dictionary.

        Args:
            traits: Dictionary with trait names and values
            name: Profile name

        Returns:
            VoiceProfile with mapped parameters
        """
        profile = VoiceProfile(name=name)

        # Direct parameter overrides
        if "gender" in traits:
            try:
                profile.gender = VoiceGender(traits["gender"])
            except ValueError:
                pass

        if "age" in traits:
            try:
                profile.age = VoiceAge(traits["age"])
            except ValueError:
                pass

        if "emotion" in traits:
            try:
                profile.default_emotion = VoiceEmotion(traits["emotion"])
            except ValueError:
                pass

        # Numeric parameters
        for param in ["pitch", "speed", "energy", "pitch_variance", "breathiness", "roughness", "warmth"]:
            if param in traits:
                value = float(traits[param])
                value = max(0.0, min(1.0, value))
                setattr(profile, param, value)

        # Apply characteristic adjustments
        if "characteristics" in traits:
            for char in traits["characteristics"]:
                char_lower = char.lower()
                if char_lower in CHARACTERISTIC_ADJUSTMENTS:
                    cls._apply_adjustments(profile, CHARACTERISTIC_ADJUSTMENTS[char_lower])

        return profile

    @classmethod
    def create_default_dm_voice(cls) -> VoiceProfile:
        """Create a default DM narration voice.

        Returns a neutral, slightly slower voice suitable for
        narrating scenes and describing environments.

        Returns:
            Default DM VoiceProfile
        """
        return VoiceProfile(
            name="default_dm",
            gender=VoiceGender.NEUTRAL,
            age=VoiceAge.ADULT,
            pitch=0.5,
            pitch_variance=0.4,
            speed=0.45,  # Slightly slower for narration clarity
            energy=0.5,
            warmth=0.55,
            default_emotion=VoiceEmotion.NEUTRAL,
        )

    @classmethod
    def create_narrator_presets(cls) -> Dict[str, VoiceProfile]:
        """Create a set of narrator voice presets.

        Returns:
            Dictionary of preset name to VoiceProfile
        """
        return {
            "epic_narrator": VoiceProfile(
                name="Epic Narrator",
                gender=VoiceGender.MALE,
                age=VoiceAge.MATURE,
                pitch=0.35,
                pitch_variance=0.5,
                speed=0.4,
                energy=0.65,
                warmth=0.5,
                default_emotion=VoiceEmotion.AUTHORITATIVE,
            ),
            "mysterious_narrator": VoiceProfile(
                name="Mysterious Narrator",
                gender=VoiceGender.NEUTRAL,
                age=VoiceAge.ADULT,
                pitch=0.45,
                pitch_variance=0.25,
                speed=0.4,
                energy=0.4,
                breathiness=0.2,
                warmth=0.35,
                default_emotion=VoiceEmotion.MYSTERIOUS,
            ),
            "friendly_narrator": VoiceProfile(
                name="Friendly Narrator",
                gender=VoiceGender.NEUTRAL,
                age=VoiceAge.ADULT,
                pitch=0.55,
                pitch_variance=0.45,
                speed=0.5,
                energy=0.6,
                warmth=0.7,
                default_emotion=VoiceEmotion.HAPPY,
            ),
            "horror_narrator": VoiceProfile(
                name="Horror Narrator",
                gender=VoiceGender.MALE,
                age=VoiceAge.MATURE,
                pitch=0.4,
                pitch_variance=0.3,
                speed=0.35,
                energy=0.35,
                breathiness=0.15,
                warmth=0.25,
                default_emotion=VoiceEmotion.OMINOUS,
            ),
        }

    @classmethod
    def _apply_adjustments(cls, profile: VoiceProfile, adjustments: Dict[str, float]) -> None:
        """Apply parameter adjustments to a profile.

        Args:
            profile: VoiceProfile to modify
            adjustments: Dictionary of parameter deltas
        """
        for param, delta in adjustments.items():
            if hasattr(profile, param):
                current = getattr(profile, param)
                new_value = max(0.0, min(1.0, current + delta))
                setattr(profile, param, new_value)

    @classmethod
    def _infer_gender(cls, description: str, npc: Any) -> Optional[VoiceGender]:
        """Infer gender from NPC description.

        Args:
            description: NPC description text
            npc: NPC object

        Returns:
            Inferred VoiceGender or None
        """
        desc_lower = description.lower()

        # Check for explicit pronouns
        male_indicators = ["he ", "him ", "his ", "himself", " man ", " male ", "gentleman", "lord ", "king ", "prince "]
        female_indicators = ["she ", "her ", "herself", " woman ", " female ", "lady ", "queen ", "princess "]

        male_count = sum(1 for ind in male_indicators if ind in desc_lower)
        female_count = sum(1 for ind in female_indicators if ind in desc_lower)

        if male_count > female_count:
            return VoiceGender.MALE
        elif female_count > male_count:
            return VoiceGender.FEMALE

        # Check NPC object for gender attribute
        gender_attr = getattr(npc, "gender", None)
        if gender_attr:
            gender_str = str(gender_attr).lower()
            if "male" in gender_str and "female" not in gender_str:
                return VoiceGender.MALE
            elif "female" in gender_str:
                return VoiceGender.FEMALE

        return None
