"""Response generation with personality application."""

import random
import re
from typing import Any, Dict, List, Optional, Tuple

from config.logging_config import get_logger
from src.personality.personality_manager import PersonalityManager, PersonalityProfile

logger = get_logger(__name__)


class ResponseGenerator:
    """Generates responses with personality traits applied."""

    def __init__(self, personality_manager: Optional[PersonalityManager] = None):
        """
        Initialize response generator.

        Args:
            personality_manager: Optional personality manager instance
        """
        self.personality_manager = personality_manager or PersonalityManager()

        # Response templates by tone
        self.tone_templates = {
            "authoritative": {
                "prefix": [
                    "According to the rules,",
                    "The regulations state that",
                    "It is mandated that",
                ],
                "suffix": [
                    "This is non-negotiable.",
                    "No exceptions are permitted.",
                    "This must be followed.",
                ],
                "connectors": ["therefore", "thus", "consequently", "hence"],
            },
            "mysterious": {
                "prefix": ["Perhaps...", "One might wonder if", "It is whispered that"],
                "suffix": [
                    "...or so the legends say.",
                    "...but who can truly know?",
                    "...shrouded in mystery.",
                ],
                "connectors": ["yet", "however", "although", "perhaps"],
            },
            "scholarly": {
                "prefix": ["Research indicates that", "Studies have shown", "Analysis reveals"],
                "suffix": [
                    "Further investigation is warranted.",
                    "This conclusion is well-supported.",
                    "The evidence is compelling.",
                ],
                "connectors": ["furthermore", "moreover", "additionally", "in addition"],
            },
            "casual": {
                "prefix": ["So basically,", "Here's the thing:", "You know what,"],
                "suffix": ["Pretty simple, right?", "That's about it.", "Hope that helps!"],
                "connectors": ["and", "but", "so", "also"],
            },
            "whimsical": {
                "prefix": ["How delightful!", "What a wonderful", "Oh, the joy of"],
                "suffix": ["How magical indeed!", "What an adventure!", "Simply enchanting!"],
                "connectors": ["and then", "meanwhile", "suddenly", "amazingly"],
            },
            "military": {
                "prefix": ["Mission objective:", "Tactical assessment:", "Strategic analysis:"],
                "suffix": [
                    "Mission parameters defined.",
                    "Proceed with caution.",
                    "Maintain operational security.",
                ],
                "connectors": ["subsequently", "following this", "next", "then"],
            },
        }

        # Perspective transformations
        self.perspective_transforms = {
            "first_person": {
                "you": "I",
                "your": "my",
                "yours": "mine",
                "yourself": "myself",
            },
            "second_person": {
                "i": "you",
                "my": "your",
                "mine": "yours",
                "myself": "yourself",
            },
            "third_person": {
                "you": "they",
                "your": "their",
                "yours": "theirs",
                "i": "one",
                "my": "one's",
            },
        }

    def generate_response(
        self,
        content: str,
        profile: Optional[PersonalityProfile] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a response with personality applied.

        Args:
            content: Base content to transform
            profile: Personality profile to apply (uses active if not provided)
            context: Optional context for response generation

        Returns:
            Personality-infused response
        """
        # Use provided profile or active profile
        if not profile:
            profile = self.personality_manager.get_active_profile()

        if not profile:
            # No personality to apply, return original
            return content

        logger.debug(f"Generating response with profile: {profile.name}")

        # Apply transformations in sequence
        response = content

        # 1. Apply perspective transformation
        response = self._apply_perspective(response, profile.perspective)

        # 2. Apply vocabulary enhancement
        response = self._apply_vocabulary(response, profile.vocabulary)

        # 3. Apply tone modifications
        response = self._apply_tone(response, profile.tone)

        # 4. Apply style adjustments
        response = self._apply_style(response, profile.style)

        # 5. Insert common phrases
        response = self._insert_phrases(response, profile.common_phrases)

        # 6. Apply sentiment adjustments
        response = self._apply_sentiment(response, profile.sentiment)

        # 7. Apply custom traits if any
        if profile.custom_traits:
            response = self._apply_custom_traits(response, profile.custom_traits)

        return response

    def _apply_perspective(self, text: str, perspective: Dict[str, Any]) -> str:
        """
        Apply perspective transformation to text.

        Args:
            text: Text to transform
            perspective: Perspective configuration

        Returns:
            Transformed text
        """
        dominant = perspective.get("dominant", "third_person")

        if dominant not in self.perspective_transforms:
            return text

        transforms = self.perspective_transforms[dominant]

        # Apply word-level transformations
        words = text.split()
        transformed_words = []

        for word in words:
            # Preserve capitalization
            lower_word = word.lower().strip(".,!?;:")
            punctuation = word[len(lower_word) :]

            if lower_word in transforms:
                # Apply transformation
                new_word = transforms[lower_word]
                # Preserve original capitalization
                if word[0].isupper():
                    new_word = new_word.capitalize()
                transformed_words.append(new_word + punctuation)
            else:
                transformed_words.append(word)

        # Handle instructional perspective
        if perspective.get("is_instructional", False):
            # Add instructional markers
            sentences = " ".join(transformed_words).split(". ")
            instructional_sentences = []

            for i, sentence in enumerate(sentences):
                if i == 0 and not sentence.startswith(("First", "To", "Step")):
                    sentence = "To proceed, " + sentence.lower()
                elif random.random() < 0.3:  # Occasionally add step markers
                    sentence = f"Next, {sentence.lower()}"

                instructional_sentences.append(sentence)

            return ". ".join(instructional_sentences)

        return " ".join(transformed_words)

    def _apply_vocabulary(self, text: str, vocabulary: Dict[str, float]) -> str:
        """
        Enhance text with characteristic vocabulary.

        Args:
            text: Text to enhance
            vocabulary: Vocabulary frequency map

        Returns:
            Enhanced text
        """
        if not vocabulary:
            return text

        # Sort vocabulary by frequency
        sorted_vocab = sorted(vocabulary.items(), key=lambda x: x[1], reverse=True)

        # Create synonym mappings for common words
        synonym_map = self._create_synonym_map(sorted_vocab[:20])  # Use top 20 words

        # Replace words with vocabulary words where appropriate
        words = text.split()
        enhanced_words = []

        for word in words:
            clean_word = word.lower().strip(".,!?;:")
            punctuation = word[len(clean_word) :]

            # Check if we have a vocabulary replacement
            if clean_word in synonym_map and random.random() < 0.3:  # 30% chance
                replacement = synonym_map[clean_word]
                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()
                enhanced_words.append(replacement + punctuation)
            else:
                enhanced_words.append(word)

        return " ".join(enhanced_words)

    def _apply_tone(self, text: str, tone: Dict[str, float]) -> str:
        """
        Apply tone modifications to text.

        Args:
            text: Text to modify
            tone: Tone characteristics

        Returns:
            Modified text
        """
        dominant_tone = tone.get("dominant", "neutral")

        if dominant_tone not in self.tone_templates:
            return text

        templates = self.tone_templates[dominant_tone]

        # Split into sentences
        sentences = text.split(". ")

        # Apply tone to sentences
        modified_sentences = []

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            # Add prefix to first sentence
            if i == 0 and random.random() < 0.5:
                prefix = random.choice(templates["prefix"])
                sentence = f"{prefix} {sentence.lower()}"

            # Add suffix to last sentence
            if i == len(sentences) - 1 and random.random() < 0.5:
                suffix = random.choice(templates["suffix"])
                sentence = f"{sentence} {suffix}"

            # Replace connectors
            for connector in ["and", "but", "so", "also"]:
                if connector in sentence.lower() and random.random() < 0.3:
                    new_connector = random.choice(templates["connectors"])
                    sentence = re.sub(
                        rf"\b{connector}\b", new_connector, sentence, flags=re.IGNORECASE
                    )

            modified_sentences.append(sentence)

        return ". ".join(modified_sentences)

    def _apply_style(self, text: str, style: Dict[str, Any]) -> str:
        """
        Apply style adjustments to text.

        Args:
            text: Text to adjust
            style: Style characteristics

        Returns:
            Adjusted text
        """
        formality = style.get("formality", 0.5)
        complexity = style.get("sentence_complexity", 0.5)
        technical = style.get("technical_level", 0.5)

        # Adjust formality
        if formality > 0.7:
            # Make more formal
            text = self._increase_formality(text)
        elif formality < 0.3:
            # Make more casual
            text = self._decrease_formality(text)

        # Adjust complexity
        if complexity > 0.7:
            # Make more complex
            text = self._increase_complexity(text)
        elif complexity < 0.3:
            # Make simpler
            text = self._simplify_sentences(text)

        # Add technical elements if needed
        if technical > 0.7:
            text = self._add_technical_elements(text)

        return text

    def _insert_phrases(self, text: str, phrases: List[str]) -> str:
        """
        Insert common phrases into text.

        Args:
            text: Text to modify
            phrases: List of common phrases

        Returns:
            Modified text
        """
        if not phrases:
            return text

        sentences = text.split(". ")

        # Occasionally insert phrases
        for i in range(len(sentences)):
            if random.random() < 0.2 and phrases:  # 20% chance
                phrase = random.choice(phrases)

                # Insert at beginning, middle, or end
                position = random.choice(["start", "middle", "end"])

                if position == "start":
                    sentences[i] = f"{phrase.capitalize()}, {sentences[i].lower()}"
                elif position == "end":
                    sentences[i] = f"{sentences[i]}, {phrase}"
                else:
                    # Insert in middle
                    words = sentences[i].split()
                    if len(words) > 3:
                        mid = len(words) // 2
                        words.insert(mid, f", {phrase},")
                        sentences[i] = " ".join(words)

        return ". ".join(sentences)

    def _apply_sentiment(self, text: str, sentiment: Dict[str, float]) -> str:
        """
        Apply sentiment adjustments to text.

        Args:
            text: Text to adjust
            sentiment: Sentiment characteristics

        Returns:
            Adjusted text
        """
        mood = sentiment.get("mood", "neutral")

        if mood == "positive":
            # Add positive modifiers
            positive_words = ["excellent", "wonderful", "great", "fantastic", "amazing"]
            text = self._insert_modifiers(text, positive_words, 0.1)
        elif mood == "negative":
            # Add negative modifiers
            negative_words = [
                "concerning",
                "problematic",
                "difficult",
                "challenging",
                "unfortunate",
            ]
            text = self._insert_modifiers(text, negative_words, 0.1)

        return text

    def _apply_custom_traits(self, text: str, custom_traits: Dict[str, Any]) -> str:
        """
        Apply custom personality traits.

        Args:
            text: Text to modify
            custom_traits: Custom trait dictionary

        Returns:
            Modified text
        """
        # Example custom traits handling
        if custom_traits.get("add_emphasis", False):
            # Add emphasis to important words
            important_words = ["must", "cannot", "always", "never", "critical", "essential"]
            for word in important_words:
                text = re.sub(rf"\b{word}\b", word.upper(), text, flags=re.IGNORECASE)

        if custom_traits.get("use_quotes", False):
            # Add occasional quotes
            sentences = text.split(". ")
            if len(sentences) > 2:
                quote_index = random.randint(1, len(sentences) - 1)
                sentences[quote_index] = f'"{sentences[quote_index]}"'
            text = ". ".join(sentences)

        return text

    def _create_synonym_map(self, vocab_words: List[Tuple[str, float]]) -> Dict[str, str]:
        """
        Create a synonym mapping for vocabulary enhancement.

        Args:
            vocab_words: List of (word, frequency) tuples

        Returns:
            Synonym mapping dictionary
        """
        # Simple synonym mappings (can be expanded)
        synonym_groups = [
            ["say", "state", "declare", "proclaim", "announce"],
            ["think", "believe", "consider", "ponder", "contemplate"],
            ["make", "create", "construct", "forge", "craft"],
            ["use", "utilize", "employ", "wield", "apply"],
            ["get", "obtain", "acquire", "procure", "secure"],
            ["give", "provide", "grant", "bestow", "offer"],
            ["show", "display", "reveal", "demonstrate", "exhibit"],
            ["find", "discover", "locate", "uncover", "detect"],
        ]

        synonym_map = {}
        vocab_word_list = [word for word, _ in vocab_words]

        for group in synonym_groups:
            # Find if any vocabulary word is in this group
            vocab_in_group = [w for w in vocab_word_list if w in group]
            if vocab_in_group:
                # Map common words to vocabulary word
                target = vocab_in_group[0]
                for word in group:
                    if word != target and word not in vocab_word_list:
                        synonym_map[word] = target

        return synonym_map

    def _increase_formality(self, text: str) -> str:
        """Increase text formality."""
        informal_to_formal = {
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "hasn't": "has not",
            "haven't": "have not",
            "shouldn't": "should not",
            "wouldn't": "would not",
            "couldn't": "could not",
            "didn't": "did not",
            "doesn't": "does not",
            "gonna": "going to",
            "wanna": "want to",
            "gotta": "have to",
            "kinda": "kind of",
            "sorta": "sort of",
            "yeah": "yes",
            "nope": "no",
            "ok": "acceptable",
            "okay": "acceptable",
        }

        for informal, formal in informal_to_formal.items():
            text = re.sub(rf"\b{informal}\b", formal, text, flags=re.IGNORECASE)

        return text

    def _decrease_formality(self, text: str) -> str:
        """Decrease text formality."""
        formal_to_informal = {
            "cannot": "can't",
            "will not": "won't",
            "do not": "don't",
            "is not": "isn't",
            "are not": "aren't",
            "was not": "wasn't",
            "were not": "weren't",
            "has not": "hasn't",
            "have not": "haven't",
            "should not": "shouldn't",
            "would not": "wouldn't",
            "could not": "couldn't",
            "did not": "didn't",
            "does not": "doesn't",
            "going to": "gonna",
            "want to": "wanna",
            "have to": "gotta",
        }

        for formal, informal in formal_to_informal.items():
            text = re.sub(rf"\b{formal}\b", informal, text, flags=re.IGNORECASE)

        return text

    def _increase_complexity(self, text: str) -> str:
        """Increase sentence complexity."""
        sentences = text.split(". ")

        # Combine some sentences
        combined = []
        i = 0
        while i < len(sentences):
            if i < len(sentences) - 1 and len(sentences[i]) < 100 and random.random() < 0.5:
                # Combine with next sentence
                connector = random.choice(["; furthermore,", ", which", ", and", "; however,"])
                combined.append(f"{sentences[i]}{connector} {sentences[i+1].lower()}")
                i += 2
            else:
                combined.append(sentences[i])
                i += 1

        return ". ".join(combined)

    def _simplify_sentences(self, text: str) -> str:
        """Simplify sentence structure."""
        # Split long sentences
        sentences = text.split(". ")
        simplified = []

        for sentence in sentences:
            if len(sentence) > 150:
                # Try to split at conjunctions
                parts = re.split(r"\s+(and|but|or|which|that)\s+", sentence)
                if len(parts) > 1:
                    # Create shorter sentences
                    for i in range(0, len(parts), 2):
                        if i < len(parts):
                            simplified.append(parts[i].capitalize())
                else:
                    simplified.append(sentence)
            else:
                simplified.append(sentence)

        return ". ".join(simplified)

    def _add_technical_elements(self, text: str) -> str:
        """Add technical elements to text."""
        # Add some technical indicators
        technical_additions = [
            "(see section 3.2)",
            "(refer to table 4-1)",
            "(as per specification)",
            "(technical note)",
            "[Technical Detail]",
        ]

        sentences = text.split(". ")

        # Add technical references occasionally
        for i in range(len(sentences)):
            if random.random() < 0.2:  # 20% chance
                addition = random.choice(technical_additions)
                sentences[i] = f"{sentences[i]} {addition}"

        return ". ".join(sentences)

    def _insert_modifiers(self, text: str, modifiers: List[str], probability: float) -> str:
        """
        Insert modifiers into text.

        Args:
            text: Text to modify
            modifiers: List of modifier words
            probability: Probability of insertion

        Returns:
            Modified text
        """
        words = text.split()
        modified_words = []

        for word in words:
            modified_words.append(word)

            # Check if next word could use a modifier
            if random.random() < probability:
                # Check if word is a noun or verb (simple heuristic)
                if not word.lower() in [
                    "the",
                    "a",
                    "an",
                    "and",
                    "or",
                    "but",
                    "in",
                    "on",
                    "at",
                    "to",
                    "for",
                ]:
                    modifier = random.choice(modifiers)
                    modified_words.insert(-1, modifier)

        return " ".join(modified_words)

    def generate_with_switching(
        self, content: str, profiles: List[PersonalityProfile], switch_probability: float = 0.3
    ) -> str:
        """
        Generate response with personality switching.

        Args:
            content: Base content
            profiles: List of profiles to switch between
            switch_probability: Probability of switching

        Returns:
            Response with mixed personalities
        """
        if not profiles:
            return content

        sentences = content.split(". ")
        result_sentences = []

        current_profile = random.choice(profiles)

        for sentence in sentences:
            # Apply current profile
            transformed = self.generate_response(sentence, current_profile)
            result_sentences.append(transformed)

            # Maybe switch profile
            if random.random() < switch_probability and len(profiles) > 1:
                # Switch to different profile
                other_profiles = [p for p in profiles if p != current_profile]
                current_profile = random.choice(other_profiles)
                logger.debug(f"Switched to profile: {current_profile.name}")

        return ". ".join(result_sentences)
