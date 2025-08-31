"""Interactive query clarification workflow for ambiguous searches."""

import difflib
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

from config.logging_config import get_logger
from src.search.error_handler import handle_search_errors

logger = get_logger(__name__)


class QueryClarificationService:
    """Service for detecting ambiguous queries and generating clarifications."""

    def __init__(self):
        """Initialize query clarification service."""
        # Ambiguous terms that often need clarification
        self.ambiguous_terms = {
            "level": ["character level", "spell level", "challenge rating level"],
            "attack": [
                "attack action",
                "attack roll",
                "attack bonus",
                "melee attack",
                "ranged attack",
            ],
            "save": ["saving throw", "save DC", "death save"],
            "action": ["action economy", "bonus action", "reaction", "standard action"],
            "damage": ["damage type", "damage roll", "damage resistance", "damage vulnerability"],
            "spell": ["spell slot", "spell level", "spell component", "spell school"],
            "class": ["character class", "armor class", "difficulty class"],
            "range": ["spell range", "weapon range", "movement range"],
            "roll": ["ability roll", "skill roll", "attack roll", "damage roll", "initiative roll"],
            "check": ["ability check", "skill check", "death check"],
        }

        # Context indicators that help disambiguate
        self.context_indicators = {
            "character": ["player", "pc", "character", "hero", "build"],
            "spell": ["cast", "magic", "cantrip", "ritual", "components"],
            "combat": ["fight", "battle", "attack", "damage", "initiative"],
            "rules": ["how to", "can i", "what happens", "rule"],
            "monster": ["creature", "enemy", "cr", "stat block"],
            "item": ["equipment", "weapon", "armor", "magic item"],
        }

        # Common query patterns that need clarification
        self.clarification_patterns = [
            (r"\b(\d+)\b", "number", "Are you looking for level {0} content, or something else?"),
            (
                r"\bhow\s+(?:do|does|to)\b",
                "instruction",
                "Do you want rules explanation or examples?",
            ),
            (
                r"\bbest\b",
                "optimization",
                "Are you looking for optimization advice or rules clarification?",
            ),
            (
                r"\bvs\.?\b|\bversus\b",
                "comparison",
                "Would you like a detailed comparison or quick differences?",
            ),
        ]

        # Track clarification history for learning
        self.clarification_history = []

    def detect_ambiguity(
        self, query: str, search_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect if a query is ambiguous and needs clarification.

        Args:
            query: The search query
            search_context: Optional context from previous searches

        Returns:
            Dictionary with ambiguity detection results
        """
        query_lower = query.lower()
        ambiguity_score = 0.0
        ambiguous_elements = []
        clarification_needed = []

        # Check for ambiguous terms
        for term, meanings in self.ambiguous_terms.items():
            if term in query_lower:
                # Check if context disambiguates
                if not self._has_disambiguating_context(query_lower, term):
                    ambiguity_score += 0.3
                    ambiguous_elements.append(
                        {"term": term, "possible_meanings": meanings, "type": "term_ambiguity"}
                    )

        # Check for pattern-based ambiguity
        for pattern, pattern_type, clarification_prompt in self.clarification_patterns:
            match = re.search(pattern, query_lower)
            if match:
                ambiguity_score += 0.2
                clarification_needed.append(
                    {
                        "pattern": pattern_type,
                        "matched_text": match.group(0),
                        "prompt": (
                            clarification_prompt.format(*match.groups())
                            if match.groups()
                            else clarification_prompt
                        ),
                    }
                )

        # Check query length (very short queries are often ambiguous)
        if len(query.split()) <= 2:
            ambiguity_score += 0.2
            clarification_needed.append(
                {
                    "type": "short_query",
                    "prompt": "Your query is quite brief. Can you provide more details?",
                }
            )

        # Check for missing context
        if search_context:
            if not search_context.get("rulebook") and not search_context.get("system"):
                ambiguity_score += 0.1
                clarification_needed.append(
                    {"type": "missing_system", "prompt": "Which game system are you asking about?"}
                )

        is_ambiguous = ambiguity_score >= 0.4

        return {
            "is_ambiguous": is_ambiguous,
            "ambiguity_score": min(ambiguity_score, 1.0),
            "ambiguous_elements": ambiguous_elements,
            "clarification_needed": clarification_needed,
            "confidence": 1.0 - min(ambiguity_score, 1.0),
        }

    def generate_clarification_questions(
        self,
        query: str,
        ambiguity_info: Dict[str, Any],
        available_content: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate clarification questions based on detected ambiguity.

        Args:
            query: Original query
            ambiguity_info: Ambiguity detection results
            available_content: Optional list of available content types

        Returns:
            List of clarification questions with options
        """
        questions = []

        # Generate questions for ambiguous terms
        for element in ambiguity_info.get("ambiguous_elements", []):
            if element["type"] == "term_ambiguity":
                question = {
                    "id": f"clarify_{element['term']}",
                    "question": f"When you say '{element['term']}', which of these did you mean?",
                    "options": element["possible_meanings"],
                    "type": "single_choice",
                    "original_term": element["term"],
                }
                questions.append(question)

        # Generate questions for patterns
        for clarification in ambiguity_info.get("clarification_needed", []):
            if clarification.get("type") == "short_query":
                question = {
                    "id": "expand_query",
                    "question": clarification["prompt"],
                    "type": "text_input",
                    "hint": "For example: 'How does sneak attack work for rogues?'",
                }
                questions.append(question)
            elif clarification.get("type") == "missing_system":
                question = {
                    "id": "select_system",
                    "question": clarification["prompt"],
                    "options": ["D&D 5e", "Pathfinder", "Call of Cthulhu", "Other"],
                    "type": "single_choice",
                }
                questions.append(question)
            elif clarification.get("pattern"):
                question = {
                    "id": f"clarify_{clarification['pattern']}",
                    "question": clarification["prompt"],
                    "type": "single_choice" if "?" in clarification["prompt"] else "text_input",
                }
                questions.append(question)

        # If we have available content, suggest specific refinements
        if available_content and len(questions) < 3:
            suggestions = self._generate_content_suggestions(query, available_content)
            if suggestions:
                question = {
                    "id": "content_suggestion",
                    "question": "Are you looking for any of these specific topics?",
                    "options": suggestions[:5],
                    "type": "multiple_choice",
                    "optional": True,
                }
                questions.append(question)

        # Limit to top 3 most relevant questions
        return questions[:3]

    def generate_query_refinements(
        self,
        query: str,
        search_results: Optional[List[Dict[str, Any]]] = None,
        user_intent: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Generate refined query suggestions.

        Args:
            query: Original query
            search_results: Optional current search results
            user_intent: Optional extracted user intent

        Returns:
            List of refined query suggestions
        """
        refinements = []
        query_lower = query.lower()

        # Add specificity if query is too general
        if len(query.split()) <= 2:
            # Add common modifiers
            modifiers = ["rules", "5e", "example", "how to use", "vs"]
            for modifier in modifiers:
                if modifier not in query_lower:
                    refinements.append(f"{query} {modifier}")

        # Add context based on intent
        if user_intent:
            content_type = user_intent.get("content_type")
            if content_type and content_type not in query_lower:
                refinements.append(f"{query} {content_type}")

            action = user_intent.get("action")
            if action == "explain" and "how" not in query_lower:
                refinements.append(f"How does {query} work")
            elif action == "list" and "all" not in query_lower:
                refinements.append(f"List all {query}")

        # Suggest corrections for common mistakes
        corrections = self._suggest_corrections(query)
        refinements.extend(corrections)

        # If we have search results, suggest related queries
        if search_results:
            related = self._extract_related_queries(query, search_results)
            refinements.extend(related)

        # Remove duplicates and return
        seen = set()
        unique_refinements = []
        for ref in refinements:
            if ref.lower() not in seen and ref.lower() != query_lower:
                seen.add(ref.lower())
                unique_refinements.append(ref)

        return unique_refinements[:5]

    def apply_clarification(
        self, original_query: str, clarification_responses: Dict[str, Any]
    ) -> str:
        """
        Apply user's clarification responses to refine the query.

        Args:
            original_query: The original ambiguous query
            clarification_responses: User's responses to clarification questions

        Returns:
            Refined query string
        """
        refined_query = original_query

        # Apply term clarifications
        for response_id, response_value in clarification_responses.items():
            if response_id.startswith("clarify_"):
                term = response_id.replace("clarify_", "")
                if isinstance(response_value, str):
                    # Replace ambiguous term with clarified version
                    refined_query = re.sub(
                        r"\b" + term + r"\b", response_value, refined_query, flags=re.IGNORECASE
                    )

            elif response_id == "expand_query":
                # User provided additional context
                if response_value and isinstance(response_value, str):
                    refined_query = f"{refined_query} {response_value}"

            elif response_id == "select_system":
                # Add system context
                if response_value and response_value not in refined_query:
                    refined_query = f"{response_value} {refined_query}"

            elif response_id == "content_suggestion":
                # Add selected content suggestions
                if isinstance(response_value, list):
                    for suggestion in response_value:
                        if suggestion not in refined_query:
                            refined_query = f"{refined_query} {suggestion}"

        # Clean up the refined query
        refined_query = re.sub(r"\s+", " ", refined_query).strip()

        # Track this clarification for learning
        self._track_clarification(original_query, clarification_responses, refined_query)

        return refined_query

    def learn_from_feedback(
        self, query: str, clarification_used: Dict[str, Any], result_quality: float
    ) -> None:
        """
        Learn from user feedback on clarification effectiveness.

        Args:
            query: Original query
            clarification_used: Clarification that was applied
            result_quality: Quality score of results (0-1)
        """
        # Store feedback for pattern learning
        feedback_entry = {
            "query": query,
            "clarification": clarification_used,
            "quality": result_quality,
            "timestamp": self._get_timestamp(),
        }

        self.clarification_history.append(feedback_entry)

        # Analyze patterns if we have enough history
        if len(self.clarification_history) >= 100:
            self._analyze_clarification_patterns()

        logger.info(f"Recorded clarification feedback: quality={result_quality}")

    def _has_disambiguating_context(self, query: str, term: str) -> bool:
        """Check if query has context that disambiguates a term."""
        for context_type, indicators in self.context_indicators.items():
            for indicator in indicators:
                if indicator in query:
                    # Check if this context helps disambiguate
                    return True
        return False

    def _generate_content_suggestions(self, query: str, available_content: List[str]) -> List[str]:
        """Generate content suggestions based on available content."""
        suggestions = []
        query_lower = query.lower()

        for content in available_content:
            # Use fuzzy matching to find related content
            similarity = difflib.SequenceMatcher(None, query_lower, content.lower()).ratio()
            if similarity > 0.3:
                suggestions.append(content)

        # Sort by relevance
        suggestions.sort(
            key=lambda x: difflib.SequenceMatcher(None, query_lower, x.lower()).ratio(),
            reverse=True,
        )

        return suggestions[:5]

    def _suggest_corrections(self, query: str) -> List[str]:
        """Suggest spelling corrections for the query."""
        corrections = []

        # Common TTRPG spelling mistakes
        common_mistakes = {
            "rouge": "rogue",
            "theif": "thief",
            "wizzard": "wizard",
            "sorceror": "sorcerer",
            "paliden": "paladin",
            "necromancy": "necromancer",
            "armour": "armor",
        }

        query_lower = query.lower()
        for mistake, correction in common_mistakes.items():
            if mistake in query_lower:
                corrected = query_lower.replace(mistake, correction)
                corrections.append(corrected)

        return corrections

    def _extract_related_queries(
        self, query: str, search_results: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract related query suggestions from search results."""
        related = []

        # Extract common terms from results
        term_frequency = Counter()
        for result in search_results[:10]:  # Only look at top results
            content = result.get("content", "").lower()
            # Extract significant terms
            words = re.findall(r"\b[a-z]{4,}\b", content)
            term_frequency.update(words)

        # Find terms that are frequent but not in original query
        query_words = set(query.lower().split())
        for term, count in term_frequency.most_common(10):
            if term not in query_words and count >= 2:
                related.append(f"{query} {term}")

        return related[:3]

    def _track_clarification(self, original: str, responses: Dict[str, Any], refined: str) -> None:
        """Track clarification for learning purposes."""
        entry = {
            "original_query": original,
            "responses": responses,
            "refined_query": refined,
            "timestamp": self._get_timestamp(),
        }

        # Keep limited history
        self.clarification_history.append(entry)
        if len(self.clarification_history) > 1000:
            self.clarification_history = self.clarification_history[-1000:]

    def _analyze_clarification_patterns(self) -> None:
        """Analyze clarification patterns to improve detection."""
        # This would analyze patterns in clarification history
        # to improve ambiguity detection over time
        successful_clarifications = [
            h for h in self.clarification_history if h.get("quality", 0) > 0.7
        ]

        if successful_clarifications:
            # Extract patterns from successful clarifications
            # This is a placeholder for more sophisticated ML analysis
            logger.info(f"Analyzed {len(successful_clarifications)} successful clarifications")

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        return datetime.utcnow().isoformat()

    @handle_search_errors()
    def get_clarification_stats(self) -> Dict[str, Any]:
        """
        Get statistics about clarification usage.

        Returns:
            Dictionary with clarification statistics
        """
        if not self.clarification_history:
            return {
                "total_clarifications": 0,
                "average_quality": 0,
                "most_ambiguous_terms": [],
                "successful_patterns": [],
            }

        total = len(self.clarification_history)
        avg_quality = sum(h.get("quality", 0) for h in self.clarification_history) / total

        # Find most common ambiguous terms
        term_counts = Counter()
        for entry in self.clarification_history:
            if "responses" in entry:
                for key in entry["responses"]:
                    if key.startswith("clarify_"):
                        term = key.replace("clarify_", "")
                        term_counts[term] += 1

        return {
            "total_clarifications": total,
            "average_quality": avg_quality,
            "most_ambiguous_terms": term_counts.most_common(5),
            "successful_patterns": self._get_successful_patterns(),
        }

    def _get_successful_patterns(self) -> List[Dict[str, Any]]:
        """Extract successful clarification patterns."""
        patterns = []

        for entry in self.clarification_history:
            if entry.get("quality", 0) > 0.8:
                patterns.append(
                    {
                        "original": entry.get("original_query", "")[:50],
                        "refined": entry.get("refined_query", "")[:50],
                        "quality": entry.get("quality", 0),
                    }
                )

        return patterns[:5]
