"""Query processing for search enhancement."""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter
import difflib
import numpy as np

from config.logging_config import get_logger
from src.pdf_processing.embedding_generator import EmbeddingGenerator
from src.core.database import ChromaDBManager

logger = get_logger(__name__)


class QueryProcessor:
    """Processes and enhances search queries."""
    
    def __init__(self, embedding_generator: Optional[EmbeddingGenerator] = None,
                 db_client: Optional[ChromaDBManager] = None):
        """Initialize query processor with optional embedding support."""
        # Initialize embedding generator and database client
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.db_client = db_client
        
        # Cache for semantic expansions to avoid repeated computations
        self.semantic_cache = {}
        
        # Common TTRPG terms for spell correction
        self.ttrpg_vocabulary = {
            # D&D terms
            "fireball", "magic missile", "cure wounds", "healing word",
            "armor class", "hit points", "saving throw", "ability check",
            "advantage", "disadvantage", "critical hit", "initiative",
            "dungeon master", "player character", "non-player character",
            "experience points", "challenge rating", "spell slot",
            "cantrip", "ritual", "concentration", "components",
            "somatic", "verbal", "material", "spell save", "spell attack",
            
            # Stats
            "strength", "dexterity", "constitution", "intelligence",
            "wisdom", "charisma", "proficiency", "expertise",
            
            # Conditions
            "blinded", "charmed", "deafened", "frightened", "grappled",
            "incapacitated", "invisible", "paralyzed", "petrified",
            "poisoned", "prone", "restrained", "stunned", "unconscious",
            
            # Damage types
            "acid", "bludgeoning", "cold", "fire", "force", "lightning",
            "necrotic", "piercing", "poison", "psychic", "radiant",
            "slashing", "thunder",
            
            # Creature types
            "aberration", "beast", "celestial", "construct", "dragon",
            "elemental", "fey", "fiend", "giant", "humanoid", "monstrosity",
            "ooze", "plant", "undead",
            
            # Actions
            "action", "bonus action", "reaction", "movement", "attack",
            "cast", "dash", "disengage", "dodge", "help", "hide", "ready",
            "search", "use",
        }
        
        # Query expansion mappings
        self.expansions = {
            "ac": ["armor class", "ac"],
            "hp": ["hit points", "hp", "health"],
            "dm": ["dungeon master", "dm", "game master", "gm"],
            "pc": ["player character", "pc", "character"],
            "npc": ["non-player character", "npc"],
            "xp": ["experience points", "xp", "exp"],
            "cr": ["challenge rating", "cr"],
            "str": ["strength", "str"],
            "dex": ["dexterity", "dex"],
            "con": ["constitution", "con"],
            "int": ["intelligence", "int"],
            "wis": ["wisdom", "wis"],
            "cha": ["charisma", "cha"],
            "save": ["saving throw", "save"],
            "dc": ["difficulty class", "dc"],
        }
        
        # Synonym groups
        self.synonyms = {
            "spell": ["spell", "magic", "incantation", "cantrip"],
            "monster": ["monster", "creature", "enemy", "foe", "beast"],
            "damage": ["damage", "harm", "hurt", "wound"],
            "heal": ["heal", "cure", "restore", "recover"],
            "attack": ["attack", "strike", "hit", "assault"],
            "defend": ["defend", "protect", "guard", "shield"],
            "wizard": ["wizard", "mage", "sorcerer", "spellcaster"],
            "warrior": ["warrior", "fighter", "soldier", "combatant"],
            "rogue": ["rogue", "thief", "assassin", "scoundrel"],
            "cleric": ["cleric", "priest", "healer", "divine"],
        }
    
    def process_query(self, query: str, use_semantic_expansion: bool = True) -> Dict[str, Any]:
        """
        Process and enhance a search query.
        
        Args:
            query: Original search query
            use_semantic_expansion: Whether to use semantic expansion
            
        Returns:
            Processed query information
        """
        # Clean and normalize
        cleaned = self.clean_query(query)
        
        # Spell correction
        corrected = self.correct_spelling(cleaned)
        
        # Expand abbreviations
        expanded = self.expand_query(corrected)
        
        # Semantic expansion (if enabled)
        semantic_expanded = expanded
        if use_semantic_expansion:
            try:
                semantic_expanded = self.expand_query_semantic(expanded)
            except Exception as e:
                logger.debug(f"Semantic expansion failed, using basic expansion: {str(e)}")
                semantic_expanded = expanded
        
        # Extract query intent
        intent = self.extract_intent(semantic_expanded)
        
        # Generate suggestions
        suggestions = self.generate_suggestions(semantic_expanded)
        
        return {
            "original": query,
            "cleaned": cleaned,
            "corrected": corrected,
            "expanded": expanded,
            "semantic_expanded": semantic_expanded,
            "intent": intent,
            "suggestions": suggestions,
        }
    
    def clean_query(self, query: str) -> str:
        """
        Clean and normalize query text.
        
        Args:
            query: Original query
            
        Returns:
            Cleaned query
        """
        # Convert to lowercase
        query = query.lower()
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query)
        
        # Remove special characters except useful ones
        query = re.sub(r'[^\w\s\+\-\']', ' ', query)
        
        # Trim
        query = query.strip()
        
        return query
    
    def correct_spelling(self, query: str) -> str:
        """
        Correct common spelling mistakes.
        
        Args:
            query: Query text
            
        Returns:
            Spell-corrected query
        """
        words = query.split()
        corrected_words = []
        
        for word in words:
            # Skip if word is already correct
            if word in self.ttrpg_vocabulary:
                corrected_words.append(word)
                continue
            
            # Find closest match
            matches = difflib.get_close_matches(
                word,
                self.ttrpg_vocabulary,
                n=1,
                cutoff=0.8
            )
            
            if matches:
                corrected_words.append(matches[0])
                logger.debug(f"Spell correction: {word} -> {matches[0]}")
            else:
                corrected_words.append(word)
        
        return " ".join(corrected_words)
    
    def expand_query(self, query: str) -> str:
        """
        Expand abbreviations and add synonyms.
        
        Args:
            query: Query text
            
        Returns:
            Expanded query
        """
        words = query.split()
        expanded_terms = []
        
        for word in words:
            # Check if it's an abbreviation
            if word in self.expansions:
                # Add expansion terms
                expanded_terms.extend(self.expansions[word])
            else:
                expanded_terms.append(word)
                
                # Add synonyms for important terms
                for syn_group in self.synonyms.values():
                    if word in syn_group:
                        # Add other synonyms from the group
                        expanded_terms.extend([s for s in syn_group if s != word])
                        break
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in expanded_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        return " ".join(unique_terms)
    
    def expand_query_semantic(self, query: str, top_k: int = 5, 
                             similarity_threshold: float = 0.7) -> str:
        """
        Expand query using semantic similarity with embeddings.
        
        Args:
            query: Query text to expand
            top_k: Number of similar terms to add
            similarity_threshold: Minimum similarity score for expansion
            
        Returns:
            Semantically expanded query
        """
        # Check cache first
        cache_key = f"{query}_{top_k}_{similarity_threshold}"
        if cache_key in self.semantic_cache:
            return self.semantic_cache[cache_key]
        
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_generator.generate_single_embedding(query)
            
            # Get vocabulary embeddings from database if available
            if self.db_client:
                # Search for semantically similar content in the database
                similar_terms = self._find_similar_terms_from_db(
                    query_embedding, top_k, similarity_threshold
                )
            else:
                # Fallback to vocabulary-based expansion
                similar_terms = self._find_similar_terms_from_vocabulary(
                    query, query_embedding, top_k, similarity_threshold
                )
            
            # Combine original query with semantic expansions
            expanded_terms = query.split() + similar_terms
            
            # Remove duplicates while preserving order
            seen = set()
            unique_terms = []
            for term in expanded_terms:
                term_lower = term.lower()
                if term_lower not in seen:
                    seen.add(term_lower)
                    unique_terms.append(term)
            
            expanded_query = " ".join(unique_terms)
            
            # Cache the result
            self.semantic_cache[cache_key] = expanded_query
            
            logger.debug(f"Semantic expansion: {query} -> {expanded_query}")
            return expanded_query
            
        except Exception as e:
            logger.error(f"Failed to perform semantic expansion: {str(e)}")
            # Fallback to original query
            return query
    
    def _find_similar_terms_from_db(self, query_embedding: List[float], 
                                   top_k: int, threshold: float) -> List[str]:
        """
        Find similar terms from database using embeddings.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of similar terms to return
            threshold: Similarity threshold
            
        Returns:
            List of similar terms
        """
        similar_terms = []
        
        try:
            # Query the database for similar content
            collections = ["rulebooks", "campaigns", "flavor_sources"]
            
            for collection_name in collections:
                try:
                    # Access collection from ChromaDBManager's collections dictionary
                    if collection_name not in self.db_client.collections:
                        logger.debug(f"Collection {collection_name} not found")
                        continue
                    collection = self.db_client.collections[collection_name]
                    
                    # Search for similar embeddings
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=top_k * 2,  # Get more results for filtering
                    )
                    
                    # Extract unique terms from results
                    if results and "documents" in results:
                        for doc in results["documents"][0]:
                            # Extract key terms from the document
                            terms = self._extract_key_terms(doc)
                            similar_terms.extend(terms)
                            
                except Exception as e:
                    logger.debug(f"Could not search collection {collection_name}: {str(e)}")
                    continue
            
            # Filter by similarity and uniqueness
            unique_terms = list(set(similar_terms))[:top_k]
            return unique_terms
            
        except Exception as e:
            logger.error(f"Database search failed: {str(e)}")
            return []
    
    def _find_similar_terms_from_vocabulary(self, query: str, query_embedding: List[float],
                                           top_k: int, threshold: float) -> List[str]:
        """
        Find similar terms from local vocabulary.
        
        Args:
            query: Original query text
            query_embedding: Query embedding vector
            top_k: Number of similar terms to return
            threshold: Similarity threshold
            
        Returns:
            List of similar terms
        """
        similar_terms = []
        
        # Generate embeddings for vocabulary terms
        vocab_terms = list(self.ttrpg_vocabulary) + \
                     [term for group in self.synonyms.values() for term in group]
        
        # Filter out terms already in query
        query_words = set(query.lower().split())
        vocab_terms = [term for term in vocab_terms if term.lower() not in query_words]
        
        # Calculate similarities
        similarities = []
        for term in vocab_terms[:100]:  # Limit to prevent performance issues
            try:
                term_embedding = self.embedding_generator.generate_single_embedding(term)
                similarity = self.embedding_generator.calculate_similarity(
                    query_embedding, term_embedding
                )
                
                if similarity >= threshold:
                    similarities.append((term, similarity))
                    
            except Exception as e:
                logger.debug(f"Could not generate embedding for {term}: {str(e)}")
                continue
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        similar_terms = [term for term, _ in similarities[:top_k]]
        
        return similar_terms
    
    def _extract_key_terms(self, text: str, max_terms: int = 5) -> List[str]:
        """
        Extract key terms from text.
        
        Args:
            text: Text to extract terms from
            max_terms: Maximum number of terms to extract
            
        Returns:
            List of key terms
        """
        # Simple keyword extraction based on TTRPG vocabulary
        words = text.lower().split()
        key_terms = []
        
        for word in words:
            # Check if word is in vocabulary or is significant
            if (word in self.ttrpg_vocabulary or 
                len(word) > 4 and word.isalpha()):
                key_terms.append(word)
                
                if len(key_terms) >= max_terms:
                    break
        
        return key_terms
    
    def extract_intent(self, query: str) -> Dict[str, Any]:
        """
        Extract the intent and context from query.
        
        Args:
            query: Query text
            
        Returns:
            Intent information
        """
        intent = {
            "type": "general",
            "content_type": None,
            "action": None,
            "modifiers": [],
        }
        
        query_lower = query.lower()
        
        # Detect content type
        if any(term in query_lower for term in ["spell", "magic", "cantrip", "incantation"]):
            intent["content_type"] = "spell"
        elif any(term in query_lower for term in ["monster", "creature", "enemy", "beast"]):
            intent["content_type"] = "monster"
        elif any(term in query_lower for term in ["rule", "rules", "mechanic", "how to"]):
            intent["content_type"] = "rule"
        elif any(term in query_lower for term in ["item", "weapon", "armor", "equipment"]):
            intent["content_type"] = "item"
        elif any(term in query_lower for term in ["class", "subclass", "archetype"]):
            intent["content_type"] = "class"
        
        # Detect action intent
        if any(term in query_lower for term in ["how to", "how do", "what is", "explain"]):
            intent["action"] = "explain"
        elif any(term in query_lower for term in ["list", "all", "show me", "find"]):
            intent["action"] = "list"
        elif any(term in query_lower for term in ["compare", "difference", "versus", "vs"]):
            intent["action"] = "compare"
        elif any(term in query_lower for term in ["create", "make", "build", "generate"]):
            intent["action"] = "create"
        
        # Detect modifiers
        if "level" in query_lower:
            intent["modifiers"].append("level-specific")
        if any(term in query_lower for term in ["quick", "summary", "brief"]):
            intent["modifiers"].append("summary")
        if any(term in query_lower for term in ["detailed", "complete", "full"]):
            intent["modifiers"].append("detailed")
        
        return intent
    
    def generate_suggestions(self, query: str) -> List[str]:
        """
        Generate query suggestions and completions.
        
        Args:
            query: Query text
            
        Returns:
            List of suggested queries
        """
        suggestions = []
        query_lower = query.lower()
        
        # Content-specific suggestions
        if "spell" in query_lower:
            base = query.replace("spell", "").strip()
            suggestions.extend([
                f"{base} spell damage",
                f"{base} spell components",
                f"{base} spell range",
                f"{base} spell duration",
                f"{base} spell level",
            ])
        
        if "monster" in query_lower or "creature" in query_lower:
            base = query.replace("monster", "").replace("creature", "").strip()
            suggestions.extend([
                f"{base} stats",
                f"{base} challenge rating",
                f"{base} abilities",
                f"{base} weaknesses",
                f"{base} loot",
            ])
        
        if "class" in query_lower:
            base = query.replace("class", "").strip()
            suggestions.extend([
                f"{base} class features",
                f"{base} class abilities",
                f"{base} class spells",
                f"{base} class build",
                f"{base} subclasses",
            ])
        
        # Rule-based suggestions
        if any(term in query_lower for term in ["how to", "how do"]):
            suggestions.extend([
                f"{query} in combat",
                f"{query} rules",
                f"{query} examples",
            ])
        
        # Filter out duplicates and empty suggestions
        suggestions = list(set(s for s in suggestions if s and s != query))
        
        # Limit suggestions
        return suggestions[:5]
    
    def fuzzy_match(self, query: str, target: str, threshold: float = 0.6) -> float:
        """
        Calculate fuzzy match score between query and target.
        
        Args:
            query: Query string
            target: Target string to match
            threshold: Minimum similarity threshold
            
        Returns:
            Similarity score (0-1)
        """
        # Use difflib for sequence matching
        matcher = difflib.SequenceMatcher(None, query.lower(), target.lower())
        similarity = matcher.ratio()
        
        # Check if above threshold
        if similarity >= threshold:
            return similarity
        
        # Also check if query is substring (partial match)
        if query.lower() in target.lower():
            # Give partial credit for substring matches
            return 0.7
        
        return 0.0
    
    def extract_filters(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract filter criteria from query.
        
        Args:
            query: Query with potential filters
            
        Returns:
            Tuple of (cleaned query, filters)
        """
        filters = {}
        cleaned_query = query
        
        # Extract level filters (e.g., "level 5 spells")
        level_match = re.search(r'level\s+(\d+)', query, re.IGNORECASE)
        if level_match:
            filters["level"] = int(level_match.group(1))
            cleaned_query = re.sub(r'level\s+\d+', '', cleaned_query, flags=re.IGNORECASE)
        
        # Extract system filters (e.g., "D&D 5e fireball")
        system_patterns = [
            (r'd&d\s*5e?', "D&D 5e"),
            (r'pathfinder', "Pathfinder"),
            (r'3\.5', "D&D 3.5"),
            (r'coc|call of cthulhu', "Call of Cthulhu"),
        ]
        
        for pattern, system in system_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                filters["system"] = system
                cleaned_query = re.sub(pattern, '', cleaned_query, flags=re.IGNORECASE)
                break
        
        # Extract source filters (e.g., "from PHB")
        source_patterns = [
            (r'phb|player\'?s? handbook', "Player's Handbook"),
            (r'dmg|dungeon master\'?s? guide', "Dungeon Master's Guide"),
            (r'mm|monster manual', "Monster Manual"),
        ]
        
        for pattern, source in source_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                filters["source"] = source
                cleaned_query = re.sub(pattern, '', cleaned_query, flags=re.IGNORECASE)
                break
        
        # Clean up extra spaces
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()
        
        return cleaned_query, filters