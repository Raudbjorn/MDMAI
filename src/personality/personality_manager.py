"""Personality profile management for TTRPG content."""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from config.logging_config import get_logger
from config.settings import settings
from src.personality.personality_extractor import PersonalityExtractor

logger = get_logger(__name__)


class PersonalityProfile:
    """Represents a personality profile for response generation."""

    def __init__(
        self,
        profile_id: str,
        name: str,
        system: str,
        tone: Dict[str, float],
        perspective: Dict[str, Any],
        style: Dict[str, Any],
        vocabulary: Dict[str, float],
        common_phrases: List[str],
        characteristics: List[str],
        sentiment: Dict[str, float] = None,
        custom_traits: Dict[str, Any] = None,
    ):
        """
        Initialize personality profile.

        Args:
            profile_id: Unique identifier
            name: Profile name
            system: Game system
            tone: Tone characteristics
            perspective: Perspective information
            style: Style patterns
            vocabulary: Key vocabulary
            common_phrases: Common phrases
            characteristics: Personality characteristics
            sentiment: Sentiment analysis
            custom_traits: Custom personality traits
        """
        self.profile_id = profile_id
        self.name = name
        self.system = system
        self.tone = tone
        self.perspective = perspective
        self.style = style
        self.vocabulary = vocabulary
        self.common_phrases = common_phrases
        self.characteristics = characteristics
        self.sentiment = sentiment or {"polarity": 0, "subjectivity": 0.5, "mood": "neutral"}
        self.custom_traits = custom_traits or {}
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.usage_count = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "system": self.system,
            "tone": self.tone,
            "perspective": self.perspective,
            "style": self.style,
            "vocabulary": self.vocabulary,
            "common_phrases": self.common_phrases,
            "characteristics": self.characteristics,
            "sentiment": self.sentiment,
            "custom_traits": self.custom_traits,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "usage_count": self.usage_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonalityProfile":
        """Create profile from dictionary."""
        profile = cls(
            profile_id=data["profile_id"],
            name=data["name"],
            system=data["system"],
            tone=data["tone"],
            perspective=data["perspective"],
            style=data["style"],
            vocabulary=data["vocabulary"],
            common_phrases=data["common_phrases"],
            characteristics=data["characteristics"],
            sentiment=data.get("sentiment"),
            custom_traits=data.get("custom_traits"),
        )

        # Restore timestamps
        if "created_at" in data:
            profile.created_at = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data:
            profile.updated_at = datetime.fromisoformat(data["updated_at"])
        if "usage_count" in data:
            profile.usage_count = data["usage_count"]

        return profile


class PersonalityManager:
    """Manages personality profiles for TTRPG content."""

    def __init__(self):
        """Initialize personality manager."""
        self.profiles_dir = settings.cache_dir / "personality_profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

        self.profiles: Dict[str, PersonalityProfile] = {}
        self.active_profile: Optional[PersonalityProfile] = None
        self.extractor = PersonalityExtractor()

        # Load existing profiles
        self._load_profiles()

        # Initialize default profiles
        self._initialize_default_profiles()

    def create_profile(
        self,
        name: str,
        system: str,
        source_text: Optional[str] = None,
        base_profile: Optional[str] = None,
        custom_traits: Optional[Dict[str, Any]] = None,
    ) -> PersonalityProfile:
        """
        Create a new personality profile.

        Args:
            name: Profile name
            system: Game system
            source_text: Optional text to extract personality from
            base_profile: Optional base profile to build upon
            custom_traits: Optional custom traits

        Returns:
            Created personality profile
        """
        logger.info(f"Creating personality profile: {name} for system: {system}")

        # Start with base profile if provided
        if base_profile and base_profile in self.profiles:
            base = self.profiles[base_profile]
            tone = base.tone.copy()
            perspective = base.perspective.copy()
            style = base.style.copy()
            vocabulary = base.vocabulary.copy()
            common_phrases = base.common_phrases.copy()
            characteristics = base.characteristics.copy()
            sentiment = base.sentiment.copy()
        else:
            # Default values
            tone = {"dominant": "neutral"}
            perspective = {"dominant": "third_person", "is_instructional": False}
            style = {"formality": 0.5, "technical_level": 0.5, "sentence_complexity": 0.5}
            vocabulary = {}
            common_phrases = []
            characteristics = []
            sentiment = {"polarity": 0, "subjectivity": 0.5, "mood": "neutral"}

        # Extract personality from source text if provided
        if source_text:
            extracted = self.extractor.extract_personality(source_text, system)
            tone = extracted["tone"]
            perspective = extracted["perspective"]
            style = extracted["style"]
            vocabulary = extracted["vocabulary"]
            common_phrases = extracted["common_phrases"]
            characteristics = extracted["characteristics"]
            sentiment = extracted["sentiment"]

        # Create profile
        profile = PersonalityProfile(
            profile_id=str(uuid.uuid4()),
            name=name,
            system=system,
            tone=tone,
            perspective=perspective,
            style=style,
            vocabulary=vocabulary,
            common_phrases=common_phrases,
            characteristics=characteristics,
            sentiment=sentiment,
            custom_traits=custom_traits or {},
        )

        # Store profile
        self.profiles[profile.profile_id] = profile
        self._save_profile(profile)

        return profile

    def get_profile(self, profile_id: str) -> Optional[PersonalityProfile]:
        """
        Get a personality profile by ID.

        Args:
            profile_id: Profile ID

        Returns:
            Profile or None if not found
        """
        return self.profiles.get(profile_id)

    def get_profile_by_name(
        self, name: str, system: Optional[str] = None
    ) -> Optional[PersonalityProfile]:
        """
        Get a profile by name and optionally system.

        Args:
            name: Profile name
            system: Optional system filter

        Returns:
            Profile or None if not found
        """
        for profile in self.profiles.values():
            if profile.name == name:
                if system is None or profile.system == system:
                    return profile
        return None

    def list_profiles(self, system: Optional[str] = None) -> List[PersonalityProfile]:
        """
        List all profiles, optionally filtered by system.

        Args:
            system: Optional system filter

        Returns:
            List of profiles
        """
        profiles = list(self.profiles.values())

        if system:
            profiles = [p for p in profiles if p.system == system]

        # Sort by usage count and name
        profiles.sort(key=lambda p: (-p.usage_count, p.name))

        return profiles

    def update_profile(
        self, profile_id: str, updates: Dict[str, Any]
    ) -> Optional[PersonalityProfile]:
        """
        Update a personality profile.

        Args:
            profile_id: Profile ID
            updates: Dictionary of updates

        Returns:
            Updated profile or None if not found
        """
        profile = self.profiles.get(profile_id)
        if not profile:
            logger.warning(f"Profile not found: {profile_id}")
            return None

        # Update allowed fields
        allowed_fields = [
            "name",
            "tone",
            "perspective",
            "style",
            "vocabulary",
            "common_phrases",
            "characteristics",
            "sentiment",
            "custom_traits",
        ]

        for field in allowed_fields:
            if field in updates:
                setattr(profile, field, updates[field])

        profile.updated_at = datetime.utcnow()

        # Save updated profile
        self._save_profile(profile)

        return profile

    def delete_profile(self, profile_id: str) -> bool:
        """
        Delete a personality profile.

        Args:
            profile_id: Profile ID

        Returns:
            True if deleted, False if not found
        """
        if profile_id not in self.profiles:
            return False

        # Remove from memory
        del self.profiles[profile_id]

        # Remove from disk
        profile_file = self.profiles_dir / f"{profile_id}.json"
        if profile_file.exists():
            profile_file.unlink()

        logger.info(f"Deleted profile: {profile_id}")
        return True

    def set_active_profile(self, profile_id: str) -> bool:
        """
        Set the active personality profile.

        Args:
            profile_id: Profile ID

        Returns:
            True if set, False if not found
        """
        profile = self.profiles.get(profile_id)
        if not profile:
            logger.warning(f"Profile not found: {profile_id}")
            return False

        self.active_profile = profile
        profile.usage_count += 1
        self._save_profile(profile)

        logger.info(f"Active profile set: {profile.name}")
        return True

    def get_active_profile(self) -> Optional[PersonalityProfile]:
        """Get the currently active profile."""
        return self.active_profile

    def merge_profiles(
        self, profile_ids: List[str], new_name: str, weights: Optional[List[float]] = None
    ) -> Optional[PersonalityProfile]:
        """
        Merge multiple profiles into a new one.

        Args:
            profile_ids: List of profile IDs to merge
            new_name: Name for the merged profile
            weights: Optional weights for each profile

        Returns:
            Merged profile or None if profiles not found
        """
        profiles = []
        for pid in profile_ids:
            profile = self.profiles.get(pid)
            if not profile:
                logger.warning(f"Profile not found for merging: {pid}")
                return None
            profiles.append(profile)

        if not profiles:
            return None

        # Use equal weights if not provided
        if not weights:
            weights = [1.0 / len(profiles)] * len(profiles)

        # Merge tone scores
        merged_tone = {}
        for profile, weight in zip(profiles, weights):
            for tone_type, score in profile.tone.items():
                if tone_type != "dominant":
                    if tone_type not in merged_tone:
                        merged_tone[tone_type] = 0
                    merged_tone[tone_type] += score * weight

        # Find dominant tone
        if merged_tone:
            dominant = max(merged_tone, key=merged_tone.get)
            merged_tone["dominant"] = dominant

        # Merge vocabulary
        merged_vocab = {}
        for profile, weight in zip(profiles, weights):
            for word, freq in profile.vocabulary.items():
                if word not in merged_vocab:
                    merged_vocab[word] = 0
                merged_vocab[word] += freq * weight

        # Combine phrases
        all_phrases = []
        for profile in profiles:
            all_phrases.extend(profile.common_phrases)

        # Combine characteristics
        all_characteristics = []
        for profile in profiles:
            all_characteristics.extend(profile.characteristics)

        # Use most common characteristics
        from collections import Counter

        char_counts = Counter(all_characteristics)
        merged_characteristics = [char for char, _ in char_counts.most_common(10)]

        # Average sentiment
        merged_sentiment = {
            "polarity": sum(p.sentiment["polarity"] * w for p, w in zip(profiles, weights)),
            "subjectivity": sum(p.sentiment["subjectivity"] * w for p, w in zip(profiles, weights)),
        }

        if merged_sentiment["polarity"] > 0.3:
            merged_sentiment["mood"] = "positive"
        elif merged_sentiment["polarity"] < -0.3:
            merged_sentiment["mood"] = "negative"
        else:
            merged_sentiment["mood"] = "neutral"

        # Use first profile's system and perspective
        system = profiles[0].system
        perspective = profiles[0].perspective
        style = profiles[0].style

        # Create merged profile
        merged_profile = PersonalityProfile(
            profile_id=str(uuid.uuid4()),
            name=new_name,
            system=system,
            tone=merged_tone,
            perspective=perspective,
            style=style,
            vocabulary=merged_vocab,
            common_phrases=list(set(all_phrases))[:20],  # Deduplicate and limit
            characteristics=merged_characteristics,
            sentiment=merged_sentiment,
            custom_traits={"merged_from": profile_ids},
        )

        # Store merged profile
        self.profiles[merged_profile.profile_id] = merged_profile
        self._save_profile(merged_profile)

        logger.info(f"Created merged profile: {new_name}")
        return merged_profile

    def _initialize_default_profiles(self):
        """Initialize default personality profiles."""
        default_profiles = [
            {
                "name": "Rules Lawyer",
                "system": "Generic",
                "tone": {
                    "authoritative": 0.8,
                    "scholarly": 0.6,
                    "formal": 0.7,
                    "dominant": "authoritative",
                },
                "perspective": {
                    "dominant": "second_person",
                    "is_instructional": True,
                },
                "style": {
                    "formality": 0.8,
                    "technical_level": 0.9,
                    "sentence_complexity": 0.6,
                },
                "vocabulary": {
                    "must": 0.05,
                    "require": 0.04,
                    "rule": 0.03,
                    "specify": 0.03,
                    "indicate": 0.02,
                },
                "common_phrases": [
                    "according to the rules",
                    "as specified in",
                    "the rules state",
                    "you must",
                    "it is required",
                ],
                "characteristics": ["precise", "technical", "authoritative", "formal"],
            },
            {
                "name": "Storyteller",
                "system": "Generic",
                "tone": {
                    "whimsical": 0.6,
                    "mysterious": 0.5,
                    "casual": 0.4,
                    "dominant": "whimsical",
                },
                "perspective": {
                    "dominant": "third_person",
                    "is_instructional": False,
                },
                "style": {
                    "formality": 0.3,
                    "technical_level": 0.2,
                    "sentence_complexity": 0.7,
                },
                "vocabulary": {
                    "adventure": 0.04,
                    "journey": 0.03,
                    "tale": 0.03,
                    "destiny": 0.02,
                    "legend": 0.02,
                },
                "common_phrases": [
                    "once upon a time",
                    "in a land",
                    "the story goes",
                    "as fate would have it",
                    "legend tells",
                ],
                "characteristics": ["narrative", "imaginative", "descriptive", "engaging"],
            },
            {
                "name": "Tactical Advisor",
                "system": "Generic",
                "tone": {
                    "military": 0.7,
                    "authoritative": 0.5,
                    "formal": 0.6,
                    "dominant": "military",
                },
                "perspective": {
                    "dominant": "second_person",
                    "is_instructional": True,
                },
                "style": {
                    "formality": 0.6,
                    "technical_level": 0.7,
                    "sentence_complexity": 0.4,
                },
                "vocabulary": {
                    "tactical": 0.05,
                    "strategic": 0.04,
                    "position": 0.03,
                    "advantage": 0.03,
                    "objective": 0.02,
                },
                "common_phrases": [
                    "tactical advantage",
                    "strategic position",
                    "primary objective",
                    "secure the area",
                    "assess the situation",
                ],
                "characteristics": ["strategic", "concise", "practical", "focused"],
            },
        ]

        for profile_data in default_profiles:
            # Check if profile already exists
            existing = self.get_profile_by_name(profile_data["name"], profile_data["system"])
            if not existing:
                # Create default profile
                profile = PersonalityProfile(
                    profile_id=str(uuid.uuid4()),
                    name=profile_data["name"],
                    system=profile_data["system"],
                    tone=profile_data["tone"],
                    perspective=profile_data["perspective"],
                    style=profile_data["style"],
                    vocabulary=profile_data["vocabulary"],
                    common_phrases=profile_data["common_phrases"],
                    characteristics=profile_data["characteristics"],
                    sentiment={"polarity": 0, "subjectivity": 0.5, "mood": "neutral"},
                    custom_traits={"is_default": True},
                )

                self.profiles[profile.profile_id] = profile
                self._save_profile(profile)

                logger.info(f"Created default profile: {profile.name}")

    def _save_profile(self, profile: PersonalityProfile):
        """Save profile to disk."""
        try:
            profile_file = self.profiles_dir / f"{profile.profile_id}.json"
            with open(profile_file, "w") as f:
                json.dump(profile.to_dict(), f, indent=2)

            logger.debug(f"Profile saved: {profile.profile_id}")

        except Exception as e:
            logger.error("Failed to save profile", error=str(e))

    def _load_profiles(self):
        """Load profiles from disk."""
        try:
            for profile_file in self.profiles_dir.glob("*.json"):
                with open(profile_file, "r") as f:
                    data = json.load(f)
                    profile = PersonalityProfile.from_dict(data)
                    self.profiles[profile.profile_id] = profile

            logger.info(f"Loaded {len(self.profiles)} personality profiles")

        except Exception as e:
            logger.error("Failed to load profiles", error=str(e))
