"""ML-based query completion and suggestion system."""

import json
import pickle
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from config.logging_config import get_logger

logger = get_logger(__name__)


class TrieNode:
    """Node for Trie data structure used in query completion."""

    def __init__(self):
        self.children = {}
        self.is_end = False
        self.frequency = 0
        self.completions = []  # Store top completions at this node


class QueryTrie:
    """Trie-based structure for efficient query completion."""

    def __init__(self):
        self.root = TrieNode()

    def insert(self, query: str, frequency: int = 1) -> None:
        """Insert a query into the trie."""
        node = self.root
        query_lower = query.lower()

        for char in query_lower:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        node.is_end = True
        node.frequency += frequency

    def search_prefix(self, prefix: str, max_results: int = 5) -> List[Tuple[str, int]]:
        """Search for queries with given prefix."""
        node = self.root
        prefix_lower = prefix.lower()

        # Navigate to prefix node
        for char in prefix_lower:
            if char not in node.children:
                return []
            node = node.children[char]

        # Collect all completions from this node
        completions = []
        self._collect_completions(node, prefix_lower, completions)

        # Sort by frequency and return top results
        completions.sort(key=lambda x: x[1], reverse=True)
        return completions[:max_results]

    def _collect_completions(
        self, node: TrieNode, prefix: str, completions: List[Tuple[str, int]]
    ) -> None:
        """Recursively collect all completions from a node."""
        if node.is_end:
            completions.append((prefix, node.frequency))

        for char, child_node in node.children.items():
            self._collect_completions(child_node, prefix + char, completions)


class QueryPatternMatcher:
    """Matches and learns query patterns for better completions."""

    def __init__(self):
        """Initialize pattern matcher."""
        # Common query patterns in TTRPG context
        self.patterns = {
            "how_to": r"^how (?:to|do|does)",
            "what_is": r"^what (?:is|are|does)",
            "rules_for": r"^rules? (?:for|about|on)",
            "spell_query": r"(?:spell|cantrip|ritual)",
            "monster_query": r"(?:monster|creature|enemy)",
            "class_query": r"(?:class|subclass|archetype)",
            "item_query": r"(?:item|weapon|armor|equipment)",
            "mechanics_query": r"(?:mechanic|system|rule)",
        }

        # Pattern-specific completions
        self.pattern_completions = {
            "how_to": [
                "how to calculate",
                "how to play",
                "how to build",
                "how to use",
                "how to create",
            ],
            "what_is": [
                "what is the difference between",
                "what is the rule for",
                "what is the best",
                "what is needed for",
            ],
            "spell_query": [
                "spell level",
                "spell components",
                "spell damage",
                "spell range",
                "spell duration",
            ],
            "monster_query": [
                "monster stats",
                "monster challenge rating",
                "monster abilities",
                "monster weaknesses",
            ],
        }

    def match_pattern(self, partial_query: str) -> Optional[str]:
        """Match query to a pattern."""
        for pattern_name, pattern_regex in self.patterns.items():
            if re.search(pattern_regex, partial_query, re.IGNORECASE):
                return pattern_name
        return None

    def get_pattern_completions(self, pattern: str, partial_query: str) -> List[str]:
        """Get completions based on matched pattern."""
        if pattern not in self.pattern_completions:
            return []

        completions = []
        for template in self.pattern_completions[pattern]:
            if template.startswith(partial_query.lower()):
                completions.append(template)

        return completions


class QueryCompletionEngine:
    """ML-based query completion engine with learning capabilities."""

    def __init__(self, model_dir: Optional[str] = None, context_window: int = 5):
        """
        Initialize query completion engine.

        Args:
            model_dir: Optional directory for saving/loading models
            context_window: Size of the context window for session context
        """
        self.model_dir = Path(model_dir) if model_dir else None
        self.lock = Lock()

        # Core data structures
        self.query_trie = QueryTrie()
        self.pattern_matcher = QueryPatternMatcher()

        # Query history and statistics
        self.query_history = []
        self.query_frequency = Counter()
        self.query_pairs = defaultdict(Counter)  # For learning query sequences
        self.term_associations = defaultdict(Counter)  # Term co-occurrence

        # Context-aware completions
        # TODO: Implement context-aware completions in future
        # self.context_completions = defaultdict(list)  # Currently unused
        self.session_context = []

        # Common TTRPG query templates
        self.query_templates = [
            "{term} rules",
            "{term} spell",
            "{term} monster",
            "{term} class",
            "how to {term}",
            "what is {term}",
            "{term} vs {term2}",
            "{term} damage",
            "{term} save",
            "{term} check",
        ]

        # Learning parameters
        self.min_frequency_threshold = 2
        self.learning_rate = 0.1
        self.context_window = context_window

        # Load existing model if available
        if self.model_dir:
            self._load_model()

    def get_completions(
        self,
        partial_query: str,
        max_suggestions: int = 5,
        use_context: bool = True,
        min_confidence: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Get query completions for partial input.

        Args:
            partial_query: Partial query string
            max_suggestions: Maximum number of suggestions
            use_context: Whether to use session context
            min_confidence: Minimum confidence threshold

        Returns:
            List of completion suggestions with metadata
        """
        with self.lock:
            suggestions = []
            partial_lower = partial_query.lower().strip()

            if not partial_lower:
                return self._get_popular_suggestions(max_suggestions)

            # 1. Trie-based prefix matching
            trie_completions = self.query_trie.search_prefix(partial_lower, max_suggestions * 2)
            for completion, frequency in trie_completions:
                confidence = self._calculate_confidence(frequency, "trie")
                if confidence >= min_confidence:
                    suggestions.append(
                        {
                            "completion": completion,
                            "type": "history",
                            "confidence": confidence,
                            "frequency": frequency,
                        }
                    )

            # 2. Pattern-based completions
            pattern = self.pattern_matcher.match_pattern(partial_lower)
            if pattern:
                pattern_completions = self.pattern_matcher.get_pattern_completions(
                    pattern, partial_lower
                )
                for completion in pattern_completions[:max_suggestions]:
                    suggestions.append(
                        {
                            "completion": completion,
                            "type": "pattern",
                            "confidence": 0.7,
                            "pattern": pattern,
                        }
                    )

            # 3. Template-based completions
            template_completions = self._generate_template_completions(partial_lower)
            suggestions.extend(template_completions[:max_suggestions])

            # 4. Context-aware completions
            if use_context and self.session_context:
                context_suggestions = self._get_context_aware_completions(partial_lower)
                suggestions.extend(context_suggestions[: max_suggestions // 2])

            # 5. Association-based completions
            association_suggestions = self._get_association_completions(partial_lower)
            suggestions.extend(association_suggestions[: max_suggestions // 2])

            # Remove duplicates and sort by confidence
            seen = set()
            unique_suggestions = []
            for sugg in suggestions:
                if sugg["completion"] not in seen:
                    seen.add(sugg["completion"])
                    unique_suggestions.append(sugg)

            unique_suggestions.sort(key=lambda x: x["confidence"], reverse=True)

            return unique_suggestions[:max_suggestions]

    def learn_from_query(self, query: str, was_successful: bool = True) -> None:
        """
        Learn from a completed query.

        Args:
            query: The completed query
            was_successful: Whether the query returned good results
        """
        with self.lock:
            query_lower = query.lower().strip()

            # Update query history
            self.query_history.append(
                {
                    "query": query_lower,
                    "timestamp": datetime.utcnow().isoformat(),
                    "successful": was_successful,
                }
            )

            # Update frequency counter
            if was_successful:
                self.query_frequency[query_lower] += 1

                # Update trie
                self.query_trie.insert(query_lower, 1)

                # Learn query pairs (for sequence prediction)
                if self.session_context:
                    last_query = self.session_context[-1]
                    self.query_pairs[last_query][query_lower] += 1

                # Update term associations
                terms = query_lower.split()
                for i, term1 in enumerate(terms):
                    for term2 in terms[i + 1 :]:
                        self.term_associations[term1][term2] += 1
                        self.term_associations[term2][term1] += 1

            # Update session context
            self.session_context.append(query_lower)
            if len(self.session_context) > self.context_window:
                self.session_context.pop(0)

            # Periodic model saving
            if len(self.query_history) % 100 == 0 and self.model_dir:
                self._save_model()

    def add_training_data(self, queries: List[str]) -> None:
        """
        Add training data to improve completions.

        Args:
            queries: List of example queries
        """
        with self.lock:
            for query in queries:
                query_lower = query.lower().strip()
                self.query_frequency[query_lower] += 1
                self.query_trie.insert(query_lower, 1)

                # Extract terms for associations
                terms = query_lower.split()
                for i, term1 in enumerate(terms):
                    for term2 in terms[i + 1 :]:
                        self.term_associations[term1][term2] += 1

            logger.info(f"Added {len(queries)} training queries")

    def get_next_word_predictions(self, partial_query: str, max_predictions: int = 3) -> List[str]:
        """
        Predict next word based on current input.

        Args:
            partial_query: Current partial query
            max_predictions: Maximum number of predictions

        Returns:
            List of predicted next words
        """
        with self.lock:
            words = partial_query.lower().split()
            if not words:
                return []

            last_word = words[-1]
            predictions = []

            # Get associated terms
            if last_word in self.term_associations:
                associated = self.term_associations[last_word].most_common(max_predictions * 2)
                for term, count in associated:
                    if term not in words:  # Avoid repeating words
                        predictions.append(term)
                        if len(predictions) >= max_predictions:
                            break

            return predictions

    def clear_session_context(self) -> None:
        """Clear the current session context."""
        with self.lock:
            self.session_context = []
            logger.debug("Session context cleared")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the completion engine.

        Returns:
            Dictionary with engine statistics
        """
        with self.lock:
            return {
                "total_queries_learned": len(self.query_history),
                "unique_queries": len(self.query_frequency),
                "most_common_queries": self.query_frequency.most_common(10),
                "total_associations": sum(len(v) for v in self.term_associations.values()),
                "session_context_size": len(self.session_context),
                "model_saved": self.model_dir is not None,
            }

    def _get_popular_suggestions(self, limit: int) -> List[Dict[str, Any]]:
        """Get popular query suggestions when no input is provided."""
        suggestions = []

        for query, frequency in self.query_frequency.most_common(limit):
            suggestions.append(
                {
                    "completion": query,
                    "type": "popular",
                    "confidence": self._calculate_confidence(frequency, "popular"),
                    "frequency": frequency,
                }
            )

        return suggestions

    def _generate_template_completions(self, partial_query: str) -> List[Dict[str, Any]]:
        """Generate completions using templates."""
        completions = []
        words = partial_query.split()

        if not words:
            return completions

        last_word = words[-1]

        for template in self.query_templates:
            # Try to match partial query to template
            if "{term}" in template:
                filled = template.replace("{term}", last_word)
                if filled.lower().startswith(partial_query):
                    completions.append(
                        {
                            "completion": filled,
                            "type": "template",
                            "confidence": 0.5,
                            "template": template,
                        }
                    )

        return completions

    def _get_context_aware_completions(self, partial_query: str) -> List[Dict[str, Any]]:
        """Get completions based on session context."""
        suggestions = []

        if not self.session_context:
            return suggestions

        # Look for query sequences
        last_query = self.session_context[-1]
        if last_query in self.query_pairs:
            for next_query, count in self.query_pairs[last_query].most_common(5):
                if next_query.startswith(partial_query):
                    suggestions.append(
                        {
                            "completion": next_query,
                            "type": "contextual",
                            "confidence": self._calculate_confidence(count, "context"),
                            "context": last_query,
                        }
                    )

        return suggestions

    def _get_association_completions(self, partial_query: str) -> List[Dict[str, Any]]:
        """Get completions based on term associations."""
        suggestions = []
        words = partial_query.split()

        if not words:
            return suggestions

        # Find queries containing associated terms
        for word in words:
            if word in self.term_associations:
                for associated_term, count in self.term_associations[word].most_common(3):
                    # Create completion with associated term
                    if associated_term not in partial_query:
                        completion = f"{partial_query} {associated_term}"
                        suggestions.append(
                            {
                                "completion": completion,
                                "type": "association",
                                "confidence": self._calculate_confidence(count, "association"),
                                "associated_term": associated_term,
                            }
                        )

        return suggestions

    def _calculate_confidence(self, frequency: int, source: str) -> float:
        """Calculate confidence score for a suggestion."""
        # Base confidence from frequency (normalize by threshold)
        FREQUENCY_NORMALIZATION = 10.0  # Frequency of 10 = 100% confidence
        base_confidence = min(frequency / FREQUENCY_NORMALIZATION, 1.0)

        # Adjust based on source
        source_weights = {
            "trie": 0.9,
            "popular": 0.8,
            "context": 0.7,
            "association": 0.6,
            "template": 0.5,
        }

        weight = source_weights.get(source, 0.5)
        return base_confidence * weight

    def _save_model(self) -> None:
        """Save the completion model to disk."""
        if not self.model_dir:
            return

        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)

            model_data = {
                "query_frequency": dict(self.query_frequency),
                "query_pairs": {k: dict(v) for k, v in self.query_pairs.items()},
                "term_associations": {k: dict(v) for k, v in self.term_associations.items()},
                "query_history": self.query_history[-1000:],  # Keep last 1000
            }

            model_file = self.model_dir / "completion_model.pkl"
            with open(model_file, "wb") as f:
                pickle.dump(model_data, f)

            logger.debug(f"Saved completion model to {model_file}")

        except Exception as e:
            logger.error(f"Failed to save completion model: {str(e)}")

    def _load_model(self) -> None:
        """Load the completion model from disk."""
        if not self.model_dir:
            return

        model_file = self.model_dir / "completion_model.pkl"
        if not model_file.exists():
            return

        try:
            with open(model_file, "r") as f:
                model_data = json.load(f)

            self.query_frequency = Counter(model_data.get("query_frequency", {}))
            self.query_pairs = defaultdict(Counter)
            for k, v in model_data.get("query_pairs", {}).items():
                self.query_pairs[k] = Counter(v)

            self.term_associations = defaultdict(Counter)
            for k, v in model_data.get("term_associations", {}).items():
                self.term_associations[k] = Counter(v)

            self.query_history = model_data.get("query_history", [])

            # Rebuild trie from frequency data
            for query, freq in self.query_frequency.items():
                self.query_trie.insert(query, freq)

            logger.info(f"Loaded completion model with {len(self.query_frequency)} unique queries")

        except Exception as e:
            logger.error(f"Failed to load completion model: {str(e)}")


class QueryCompletionService:
    """High-level service for query completion functionality."""

    def __init__(self, model_dir: Optional[str] = None, context_window: int = 5):
        """
        Initialize query completion service.

        Args:
            model_dir: Optional directory for model persistence
            context_window: Size of the context window for session context
        """
        self.engine = QueryCompletionEngine(model_dir, context_window)

        # Pre-load common TTRPG queries for better initial suggestions
        self._load_default_queries()

    def get_suggestions(self, partial_query: str, max_suggestions: int = 5) -> List[Dict[str, Any]]:
        """
        Get query completion suggestions.

        Args:
            partial_query: Partial query string
            max_suggestions: Maximum number of suggestions

        Returns:
            List of suggestions with metadata
        """
        return self.engine.get_completions(partial_query, max_suggestions)

    def learn(self, query: str, successful: bool = True) -> None:
        """
        Learn from a completed query.

        Args:
            query: Completed query
            successful: Whether query was successful
        """
        self.engine.learn_from_query(query, successful)

    def predict_next_word(self, partial_query: str) -> List[str]:
        """
        Predict the next word in a query.

        Args:
            partial_query: Current partial query

        Returns:
            List of predicted next words
        """
        return self.engine.get_next_word_predictions(partial_query)

    def clear_context(self) -> None:
        """Clear session context."""
        self.engine.clear_session_context()

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return self.engine.get_statistics()

    def _load_default_queries(self) -> None:
        """Load default TTRPG queries for initial training."""
        default_queries = [
            # D&D 5e common queries
            "fireball spell damage",
            "how to calculate armor class",
            "sneak attack rules",
            "advantage and disadvantage",
            "death saving throws",
            "spell slots per level",
            "multiclassing requirements",
            "concentration rules",
            "opportunity attack rules",
            "critical hit damage",
            # General TTRPG queries
            "character creation guide",
            "how to run combat",
            "initiative order",
            "skill check difficulty",
            "experience points calculation",
            "treasure generation",
            "random encounter tables",
            "npc generation",
            # Rules queries
            "grappling rules",
            "mounted combat rules",
            "underwater combat",
            "falling damage",
            "resting rules",
            "exhaustion levels",
            "conditions list",
            # Class queries
            "wizard spell list",
            "fighter subclasses",
            "rogue abilities",
            "cleric domains",
            "barbarian rage",
            "paladin oaths",
            "ranger favored enemy",
            # Monster queries
            "dragon challenge rating",
            "undead immunities",
            "monster manual",
            "creature types",
            "legendary actions",
            "lair actions",
        ]

        self.engine.add_training_data(default_queries)
