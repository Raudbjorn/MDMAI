"""Personality extraction from TTRPG source materials."""

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import spacy
from textblob import TextBlob

from config.logging_config import get_logger

logger = get_logger(__name__)


class PersonalityExtractor:
    """Extracts personality characteristics from text."""

    def __init__(self):
        """Initialize personality extractor."""
        try:
            # Load spaCy model for NLP
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, using basic extraction")
            self.nlp = None

        # Tone indicators
        self.tone_indicators = {
            "authoritative": [
                "must",
                "shall",
                "will",
                "require",
                "mandate",
                "command",
                "dictate",
                "decree",
                "ordain",
                "it is written",
                "the law states",
            ],
            "mysterious": [
                "perhaps",
                "maybe",
                "unknown",
                "hidden",
                "secret",
                "enigma",
                "whisper",
                "shadow",
                "veil",
                "obscure",
                "cryptic",
                "arcane",
            ],
            "scholarly": [
                "research",
                "study",
                "examine",
                "analyze",
                "hypothesis",
                "theory",
                "evidence",
                "conclude",
                "observe",
                "document",
                "according to",
                "it has been noted",
            ],
            "casual": [
                "gonna",
                "wanna",
                "yeah",
                "nope",
                "stuff",
                "thing",
                "basically",
                "kinda",
                "sorta",
                "like",
                "you know",
            ],
            "formal": [
                "therefore",
                "thus",
                "hence",
                "moreover",
                "furthermore",
                "consequently",
                "accordingly",
                "whereas",
                "hereby",
                "thereof",
            ],
            "ominous": [
                "doom",
                "dread",
                "fear",
                "terror",
                "horror",
                "nightmare",
                "darkness",
                "evil",
                "curse",
                "forbidden",
                "unspeakable",
            ],
            "whimsical": [
                "delightful",
                "charming",
                "enchanting",
                "magical",
                "wonder",
                "sparkle",
                "dance",
                "sing",
                "laugh",
                "joy",
                "merry",
            ],
            "military": [
                "mission",
                "objective",
                "target",
                "operation",
                "tactical",
                "strategic",
                "command",
                "protocol",
                "classified",
                "secure",
            ],
        }

        # Perspective indicators
        self.perspective_indicators = {
            "first_person": ["i", "me", "my", "mine", "we", "us", "our"],
            "second_person": ["you", "your", "yours"],
            "third_person": ["he", "she", "it", "they", "them", "their"],
            "omniscient": ["all-knowing", "it is known", "the truth is", "in reality"],
            "instructional": ["step", "first", "next", "then", "finally", "to do this"],
        }

        # Style patterns
        self.style_patterns = {
            "descriptive": r"\b(is|are|was|were|seems?|appears?|looks?)\s+\w+ly\b",
            "action_oriented": r"\b(attack|defend|cast|move|run|fight|strike)\b",
            "technical": r"\b\d+d\d+\b|\b\d+[+-]\d+\b|DC\s*\d+",
            "narrative": r"\b(once upon|long ago|in the|there was|story|tale)\b",
        }

    def extract_personality(self, text: str, system: str = "Unknown") -> Dict[str, Any]:
        """
        Extract personality characteristics from text.

        Args:
            text: Source text to analyze
            system: Game system name

        Returns:
            Personality profile dictionary
        """
        logger.info(f"Extracting personality for system: {system}")

        # Clean and prepare text
        text_lower = text.lower()
        sentences = self._split_sentences(text)

        # Extract various characteristics
        tone = self._detect_tone(text_lower)
        perspective = self._detect_perspective(text_lower)
        style = self._analyze_style(text)
        vocabulary = self._extract_vocabulary(text)
        common_phrases = self._extract_common_phrases(sentences)
        sentiment = self._analyze_sentiment(text)

        # Build personality profile
        profile = {
            "system": system,
            "tone": tone,
            "perspective": perspective,
            "style": style,
            "vocabulary": vocabulary,
            "common_phrases": common_phrases,
            "sentiment": sentiment,
            "characteristics": self._determine_characteristics(tone, style, sentiment),
        }

        return profile

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if self.nlp:
            doc = self.nlp(text[:1000000])  # Limit for performance
            return [sent.text.strip() for sent in doc.sents]
        else:
            # Basic sentence splitting
            sentences = re.split(r"[.!?]+", text)
            return [s.strip() for s in sentences if s.strip()]

    def _detect_tone(self, text: str) -> Dict[str, float]:
        """
        Detect tone characteristics in text.

        Args:
            text: Lowercase text

        Returns:
            Tone scores
        """
        tone_scores = {}
        text_words = text.split()
        text_length = len(text_words)

        for tone, indicators in self.tone_indicators.items():
            # Count occurrences of tone indicators
            count = sum(1 for word in text_words if word in indicators)

            # Also check for phrases
            for indicator in indicators:
                if " " in indicator:  # Multi-word phrase
                    count += text.count(indicator)

            # Normalize by text length
            score = count / text_length if text_length > 0 else 0
            tone_scores[tone] = min(score * 100, 1.0)  # Scale and cap at 1.0

        # Find dominant tone
        if tone_scores:
            max_tone = max(tone_scores, key=tone_scores.get)
            tone_scores["dominant"] = max_tone

        return tone_scores

    def _detect_perspective(self, text: str) -> Dict[str, Any]:
        """
        Detect narrative perspective.

        Args:
            text: Lowercase text

        Returns:
            Perspective information
        """
        perspective_counts = defaultdict(int)
        words = text.split()

        for perspective, indicators in self.perspective_indicators.items():
            for word in words:
                if word in indicators:
                    perspective_counts[perspective] += 1

        # Determine dominant perspective
        if perspective_counts:
            dominant = max(perspective_counts, key=perspective_counts.get)
        else:
            dominant = "third_person"  # Default

        # Check for instructional style
        instructional_score = (
            len(re.findall(r"\b(step \d+|first,|then,|finally,)", text)) / len(words)
            if words
            else 0
        )

        return {
            "dominant": dominant,
            "scores": dict(perspective_counts),
            "is_instructional": instructional_score > 0.01,
        }

    def _analyze_style(self, text: str) -> Dict[str, Any]:
        """
        Analyze writing style.

        Args:
            text: Source text

        Returns:
            Style characteristics
        """
        style_info = {
            "patterns": {},
            "sentence_complexity": 0,
            "formality": 0,
            "technical_level": 0,
        }

        # Check style patterns
        for style, pattern in self.style_patterns.items():
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            style_info["patterns"][style] = matches

        # Analyze sentence complexity
        sentences = self._split_sentences(text)
        if sentences:
            avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
            style_info["sentence_complexity"] = min(avg_length / 30, 1.0)  # Normalize

        # Formality score
        formal_words = len(
            re.findall(r"\b(therefore|thus|hence|moreover|furthermore)\b", text.lower())
        )
        casual_words = len(re.findall(r"\b(gonna|wanna|yeah|stuff|thing)\b", text.lower()))

        if formal_words + casual_words > 0:
            style_info["formality"] = formal_words / (formal_words + casual_words)

        # Technical level (game mechanics)
        technical_matches = len(re.findall(r"\b\d+d\d+\b|DC\s*\d+|\+\d+\s+bonus", text))
        style_info["technical_level"] = min(technical_matches / 100, 1.0)

        return style_info

    def _extract_vocabulary(self, text: str) -> Dict[str, float]:
        """
        Extract characteristic vocabulary.

        Args:
            text: Source text

        Returns:
            Vocabulary frequency map
        """
        # Tokenize and clean
        words = re.findall(r"\b[a-z]+\b", text.lower())

        # Filter common words
        common_words = {
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
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "along",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
        }

        # Count frequencies
        word_counts = Counter(w for w in words if w not in common_words and len(w) > 3)

        # Get top vocabulary
        total_words = sum(word_counts.values())
        vocabulary = {}

        for word, count in word_counts.most_common(100):
            frequency = count / total_words if total_words > 0 else 0
            vocabulary[word] = frequency

        return vocabulary

    def _extract_common_phrases(self, sentences: List[str]) -> List[str]:
        """
        Extract common phrases and expressions.

        Args:
            sentences: List of sentences

        Returns:
            List of common phrases
        """
        phrases = []

        # Look for recurring multi-word patterns
        bigrams = []
        trigrams = []

        for sentence in sentences:
            words = sentence.lower().split()

            # Extract bigrams
            for i in range(len(words) - 1):
                bigrams.append(f"{words[i]} {words[i+1]}")

            # Extract trigrams
            for i in range(len(words) - 2):
                trigrams.append(f"{words[i]} {words[i+1]} {words[i+2]}")

        # Count frequencies
        bigram_counts = Counter(bigrams)
        trigram_counts = Counter(trigrams)

        # Get common phrases (appearing more than once)
        for phrase, count in bigram_counts.most_common(20):
            if count > 2 and not any(
                word in ["the", "a", "an", "is", "are"] for word in phrase.split()
            ):
                phrases.append(phrase)

        for phrase, count in trigram_counts.most_common(10):
            if count > 2:
                phrases.append(phrase)

        return phrases[:20]  # Limit to top 20

    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze overall sentiment and mood.

        Args:
            text: Source text

        Returns:
            Sentiment scores
        """
        try:
            blob = TextBlob(text[:5000])  # Limit for performance

            sentiment = {
                "polarity": blob.sentiment.polarity,  # -1 to 1
                "subjectivity": blob.sentiment.subjectivity,  # 0 to 1
            }

            # Classify mood
            if sentiment["polarity"] > 0.3:
                sentiment["mood"] = "positive"
            elif sentiment["polarity"] < -0.3:
                sentiment["mood"] = "negative"
            else:
                sentiment["mood"] = "neutral"

            return sentiment

        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return {
                "polarity": 0,
                "subjectivity": 0.5,
                "mood": "neutral",
            }

    def _determine_characteristics(
        self, tone: Dict[str, float], style: Dict[str, Any], sentiment: Dict[str, float]
    ) -> List[str]:
        """
        Determine overall personality characteristics.

        Args:
            tone: Tone analysis
            style: Style analysis
            sentiment: Sentiment analysis

        Returns:
            List of characteristics
        """
        characteristics = []

        # Based on dominant tone
        if "dominant" in tone:
            dominant_tone = tone["dominant"]
            if dominant_tone == "authoritative":
                characteristics.append("commanding")
            elif dominant_tone == "mysterious":
                characteristics.append("enigmatic")
            elif dominant_tone == "scholarly":
                characteristics.append("academic")
            elif dominant_tone == "ominous":
                characteristics.append("foreboding")

        # Based on formality
        if style["formality"] > 0.7:
            characteristics.append("formal")
        elif style["formality"] < 0.3:
            characteristics.append("casual")

        # Based on technicality
        if style["technical_level"] > 0.5:
            characteristics.append("technical")

        # Based on sentiment
        if sentiment["mood"] == "positive":
            characteristics.append("optimistic")
        elif sentiment["mood"] == "negative":
            characteristics.append("pessimistic")

        # Based on sentence complexity
        if style["sentence_complexity"] > 0.7:
            characteristics.append("elaborate")
        elif style["sentence_complexity"] < 0.3:
            characteristics.append("concise")

        return characteristics
