"""Adaptive learning system for improving PDF processing over time."""

import json
import os
import pickle
import re
import tempfile
from collections import defaultdict
from typing import Any, Dict, Optional

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class PatternTemplate:
    """Represents a learned pattern for content extraction."""

    def __init__(self, pattern_type: str, system: str):
        """
        Initialize a pattern template.

        Args:
            pattern_type: Type of pattern (e.g., 'spell', 'monster', 'rule')
            system: Game system (e.g., 'D&D 5e', 'Pathfinder')
        """
        self.pattern_type = pattern_type
        self.system = system
        self.field_patterns = {}  # Field name -> regex patterns
        self.field_positions = {}  # Field name -> typical position
        self.confidence_scores = defaultdict(float)
        self.usage_count = 0
        self.success_rate = 1.0

    def add_example(self, text: str, fields: Dict[str, str]):
        """
        Learn from an example of this pattern type.

        Args:
            text: Example text
            fields: Extracted fields from the text
        """
        self.usage_count += 1

        for field_name, field_value in fields.items():
            if field_value in text:
                # Find the pattern that captures this field
                position = text.index(field_value)

                # Store position information
                if field_name not in self.field_positions:
                    self.field_positions[field_name] = []
                self.field_positions[field_name].append(position / len(text))

                # Try to extract a pattern
                pattern = self._extract_field_pattern(text, field_value, field_name)
                if pattern:
                    if field_name not in self.field_patterns:
                        self.field_patterns[field_name] = []
                    self.field_patterns[field_name].append(pattern)

    def _extract_field_pattern(self, text: str, value: str, field_name: str) -> Optional[str]:
        """
        Extract a regex pattern for finding a field.

        Args:
            text: Full text
            value: Field value
            field_name: Name of the field

        Returns:
            Regex pattern or None
        """
        # Common patterns for different field types
        if field_name in ["casting_time", "range", "duration"]:
            return rf'{field_name.replace("_", " ").title()}:\s*([^,\n]+)'
        elif field_name in ["ac", "hp", "speed"]:
            return rf"{field_name.upper()}:?\s*(\d+)"
        elif field_name == "challenge_rating":
            return r"Challenge:?\s*(\d+(?:/\d+)?)\s*\([0-9,]+ XP\)"
        else:
            # Generic pattern
            return rf'{field_name.replace("_", " ").title()}:?\s*([^,\n]+)'

    def apply(self, text: str) -> Dict[str, Any]:
        """
        Apply this pattern to extract fields from text.

        Args:
            text: Text to process

        Returns:
            Extracted fields
        """
        results = {}

        for field_name, patterns in self.field_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    results[field_name] = match.group(1).strip()
                    break

        return results


class AdaptiveLearningSystem:
    """Learns and improves content extraction patterns over time."""

    def __init__(self):
        """Initialize the adaptive learning system."""
        self.cache_dir = settings.cache_dir / "adaptive_learning"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.patterns = defaultdict(dict)  # system -> pattern_type -> PatternTemplate
        self.content_classifiers = {}  # Trained classifiers for content types
        self.extraction_metrics = defaultdict(lambda: {"success": 0, "total": 0})

        self._load_cached_patterns()

    def learn_from_document(
        self,
        pdf_content: Dict[str, Any],
        system: str,
        manual_corrections: Optional[Dict[str, Any]] = None,
    ):
        """
        Learn patterns from a processed document.

        Args:
            pdf_content: Extracted PDF content
            system: Game system
            manual_corrections: Optional manual corrections to learn from
        """
        logger.info(f"Learning from document for system: {system}")

        # Learn content type patterns
        self._learn_content_types(pdf_content, system)

        # Learn extraction patterns
        self._learn_extraction_patterns(pdf_content, system)

        # Apply manual corrections if provided
        if manual_corrections:
            self._apply_corrections(manual_corrections, system)

        # Save updated patterns
        self._save_patterns()

    def _learn_content_types(self, pdf_content: Dict[str, Any], system: str):
        """
        Learn to classify content types.

        Args:
            pdf_content: Extracted content
            system: Game system
        """
        # Extract features from content blocks
        for page in pdf_content.get("pages", []):
            text = page.get("text", "")

            # Extract various features
            features = self._extract_content_features(text)

            # Classify based on features
            content_type = self._classify_by_features(features)

            # Store classification pattern
            if system not in self.content_classifiers:
                self.content_classifiers[system] = defaultdict(list)

            self.content_classifiers[system][content_type].append(features)

    def _extract_content_features(self, text: str) -> Dict[str, Any]:
        """
        Extract features for content classification.

        Args:
            text: Content text

        Returns:
            Feature dictionary
        """
        features = {
            "length": len(text),
            "num_lines": text.count("\n"),
            "num_digits": sum(c.isdigit() for c in text),
            "num_uppercase": sum(c.isupper() for c in text),
            "has_table_chars": "|" in text or "\t" in text,
            "has_dice_notation": bool(re.search(r"\d+d\d+", text)),
            "has_stats": bool(re.search(r"\b(STR|DEX|CON|INT|WIS|CHA)\b", text)),
            "has_spell_keywords": any(
                kw in text.lower() for kw in ["casting time", "components", "duration", "spell"]
            ),
            "has_monster_keywords": any(
                kw in text.lower() for kw in ["hit points", "armor class", "challenge", "creature"]
            ),
            "has_rule_keywords": any(
                kw in text.lower()
                for kw in ["must", "cannot", "action", "bonus action", "reaction"]
            ),
        }

        return features

    def _classify_by_features(self, features: Dict[str, Any]) -> str:
        """
        Classify content type based on features.

        Args:
            features: Extracted features

        Returns:
            Content type classification
        """
        # Simple rule-based classification (can be replaced with ML model)
        if features["has_spell_keywords"]:
            return "spell"
        elif features["has_monster_keywords"]:
            return "monster"
        elif features["has_stats"] and features["has_dice_notation"]:
            return "stat_block"
        elif features["has_table_chars"]:
            return "table"
        elif features["has_rule_keywords"]:
            return "rule"
        else:
            return "narrative"

    def _learn_extraction_patterns(self, pdf_content: Dict[str, Any], system: str):
        """
        Learn patterns for extracting structured data.

        Args:
            pdf_content: Extracted content
            system: Game system
        """
        for page in pdf_content.get("pages", []):
            text = page.get("text", "")

            # Try to identify and learn from known structures
            if "spell" in text.lower():
                self._learn_spell_pattern(text, system)
            if "challenge" in text.lower() or "creature" in text.lower():
                self._learn_monster_pattern(text, system)

    def _learn_spell_pattern(self, text: str, system: str):
        """
        Learn spell format patterns.

        Args:
            text: Text containing spell
            system: Game system
        """
        # Look for common spell fields
        fields = {}

        # Casting time
        match = re.search(r"Casting Time:\s*([^\n]+)", text, re.IGNORECASE)
        if match:
            fields["casting_time"] = match.group(1)

        # Range
        match = re.search(r"Range:\s*([^\n]+)", text, re.IGNORECASE)
        if match:
            fields["range"] = match.group(1)

        # Duration
        match = re.search(r"Duration:\s*([^\n]+)", text, re.IGNORECASE)
        if match:
            fields["duration"] = match.group(1)

        # Components
        match = re.search(r"Components:\s*([^\n]+)", text, re.IGNORECASE)
        if match:
            fields["components"] = match.group(1)

        if fields:
            # Get or create pattern template
            if "spell" not in self.patterns[system]:
                self.patterns[system]["spell"] = PatternTemplate("spell", system)

            self.patterns[system]["spell"].add_example(text, fields)

    def _learn_monster_pattern(self, text: str, system: str):
        """
        Learn monster/creature format patterns.

        Args:
            text: Text containing monster stats
            system: Game system
        """
        fields = {}

        # Armor Class
        match = re.search(r"Armor Class:?\s*(\d+)", text, re.IGNORECASE)
        if match:
            fields["armor_class"] = match.group(1)

        # Hit Points
        match = re.search(r"Hit Points:?\s*(\d+)", text, re.IGNORECASE)
        if match:
            fields["hit_points"] = match.group(1)

        # Challenge Rating
        match = re.search(r"Challenge:?\s*(\d+(?:/\d+)?)", text, re.IGNORECASE)
        if match:
            fields["challenge_rating"] = match.group(1)

        # Stats
        stat_pattern = (
            r"STR\s*(\d+).*?DEX\s*(\d+).*?CON\s*(\d+).*?INT\s*(\d+).*?WIS\s*(\d+).*?CHA\s*(\d+)"
        )
        match = re.search(stat_pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            # Flatten stats into individual fields for proper processing
            fields["str"] = match.group(1)
            fields["dex"] = match.group(2)
            fields["con"] = match.group(3)
            fields["int"] = match.group(4)
            fields["wis"] = match.group(5)
        # Stats (extract each independently to handle any order)
        for stat in ["str", "dex", "con", "int", "wis", "cha"]:
            stat_pattern = rf"{stat.upper()}\s*(\d+)"
            match = re.search(stat_pattern, text, re.IGNORECASE)
            if match:
                fields[stat] = match.group(1)

        if fields:
            # Get or create pattern template
            if "monster" not in self.patterns[system]:
                self.patterns[system]["monster"] = PatternTemplate("monster", system)

            self.patterns[system]["monster"].add_example(text, fields)

    def apply_learned_patterns(
        self, text: str, system: str, content_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Apply learned patterns to extract structured data.

        Args:
            text: Text to process
            system: Game system
            content_type: Optional content type hint

        Returns:
            Extracted structured data
        """
        results = {}

        # Determine content type if not provided
        if not content_type:
            features = self._extract_content_features(text)
            content_type = self._classify_by_features(features)

        # Apply appropriate pattern
        if system in self.patterns and content_type in self.patterns[system]:
            pattern = self.patterns[system][content_type]
            extracted = pattern.apply(text)
            if extracted:
                results = {
                    "type": content_type,
                    "fields": extracted,
                    "confidence": pattern.success_rate,
                }

        # Track metrics
        self.extraction_metrics[f"{system}_{content_type}"]["total"] += 1
        if results:
            self.extraction_metrics[f"{system}_{content_type}"]["success"] += 1

        return results

    def _apply_corrections(self, corrections: Dict[str, Any], system: str):
        """
        Learn from manual corrections.

        Args:
            corrections: Manual corrections
            system: Game system
        """
        for content_type, examples in corrections.items():
            if content_type not in self.patterns[system]:
                self.patterns[system][content_type] = PatternTemplate(content_type, system)

            for example in examples:
                self.patterns[system][content_type].add_example(example["text"], example["fields"])

    def get_extraction_stats(self) -> Dict[str, Any]:
        """
        Get statistics about extraction performance.

        Returns:
            Extraction statistics
        """
        stats = {}

        for key, metrics in self.extraction_metrics.items():
            success_rate = metrics["success"] / metrics["total"] if metrics["total"] > 0 else 0
            stats[key] = {
                "success_rate": success_rate,
                "total_attempts": metrics["total"],
                "successful_extractions": metrics["success"],
            }

        return stats

    def _atomic_save(self, data: Any, file_name: str, is_json: bool = False) -> None:
        """
        Atomically save data to a file.

        Args:
            data: Data to save
            file_name: Name of the file to save to
            is_json: Whether to save as JSON (True) or pickle (False)

        Raises:
            Exception: If save fails
        """
        file_path = self.cache_dir / file_name
        mode = "w" if is_json else "wb"

        tmp_file = None
        tmp_name = None
        try:
            with tempfile.NamedTemporaryFile(mode=mode, delete=False, dir=self.cache_dir) as tmp:
                tmp_file = tmp
                tmp_name = tmp.name  # Keep reference for cleanup
                if is_json:
                    json.dump(data, tmp)
                else:
                    pickle.dump(data, tmp)
                tmp.flush()
                os.fsync(tmp.fileno())

            # Atomic replace
            os.replace(tmp_name, file_path)
            tmp_name = None  # Clear since file was successfully moved

        except Exception as e:
            # Clean up temporary file if it still exists
            if tmp_name and os.path.exists(tmp_name):
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass  # Best effort cleanup
            raise e

    def _save_patterns(self):
        """Save learned patterns to cache with atomic operations."""
        try:
            # Save patterns
            self._atomic_save(dict(self.patterns), "patterns.pkl")

            # Save classifiers
            self._atomic_save(self.content_classifiers, "classifiers.pkl")

            # Save metrics
            self._atomic_save(dict(self.extraction_metrics), "metrics.json", is_json=True)

            logger.debug("Adaptive learning patterns saved atomically")

        except Exception as e:
            logger.error(f"Failed to save patterns: {str(e)}")

    def _load_cached_patterns(self):
        """Load previously learned patterns from cache with validation."""
        try:
            # Load patterns with validation
            patterns_file = self.cache_dir / "patterns.pkl"
            if patterns_file.exists():
                try:
                    with open(patterns_file, "rb") as f:
                        loaded_patterns = pickle.load(f)
                        if isinstance(loaded_patterns, dict):
                            self.patterns.update(loaded_patterns)
                        else:
                            logger.warning("Invalid patterns file format, skipping")
                except (pickle.UnpicklingError, EOFError) as e:
                    logger.error(f"Corrupted patterns file: {e}")

            # Load classifiers with validation
            classifiers_file = self.cache_dir / "classifiers.pkl"
            if classifiers_file.exists():
                try:
                    with open(classifiers_file, "rb") as f:
                        loaded_classifiers = pickle.load(f)
                        if isinstance(loaded_classifiers, dict):
                            self.content_classifiers = loaded_classifiers
                        else:
                            logger.warning("Invalid classifiers file format, skipping")
                except (pickle.UnpicklingError, EOFError) as e:
                    logger.error(f"Corrupted classifiers file: {e}")

            # Load metrics
            metrics_file = self.cache_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    loaded_metrics = json.load(f)
                    for key, value in loaded_metrics.items():
                        self.extraction_metrics[key] = value

            logger.info(
                "Loaded adaptive learning cache",
                num_systems=len(self.patterns),
                num_classifiers=len(self.content_classifiers),
            )

        except Exception as e:
            logger.warning("Could not load cached patterns", error=str(e))
