"""Campaign-Rulebook linking system for contextual search."""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from config.logging_config import get_logger
from src.campaign.models import Campaign
from src.core.database import ChromaDBManager
from src.search.error_handler import handle_search_errors

logger = get_logger(__name__)


@dataclass
class RulebookReference:
    """Represents a reference to rulebook content."""

    source_type: str  # "character", "npc", "location", "plot_point"
    source_id: str  # ID of the source entity
    source_name: str  # Name of the source entity
    rulebook_id: str  # ID of the rulebook document
    rulebook_name: str  # Name of the rulebook
    page: Optional[int] = None
    section: Optional[str] = None
    content_type: str = ""  # "rule", "spell", "monster", etc.
    reference_text: str = ""  # The actual referenced text
    confidence: float = 1.0  # Confidence score of the link

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "rulebook_id": self.rulebook_id,
            "rulebook_name": self.rulebook_name,
            "page": self.page,
            "section": self.section,
            "content_type": self.content_type,
            "reference_text": self.reference_text,
            "confidence": self.confidence,
        }


class RulebookLinker:
    """Manages links between campaign content and rulebook references."""

    # Default similarity threshold for matching
    DEFAULT_SIMILARITY_THRESHOLD = 0.7

    # Compiled regex patterns for detecting rulebook references
    REFERENCE_PATTERNS = [
        re.compile(
            r"\b(?:see|ref|reference|per|according to|as per|following)\s+([A-Z][^,.;!?]{2,30})",
            re.IGNORECASE,
        ),
        re.compile(r"\b(?:PHB|DMG|MM|XGE|TCE|VGM|MTF)\s*(?:p\.?|page)?\s*(\d+)", re.IGNORECASE),
        re.compile(r"\(([A-Z][^)]{2,30})\)"),
        re.compile(r"(?:spell|ability|feat|class feature|trait):\s*([A-Za-z\s]+)", re.IGNORECASE),
        re.compile(r"(?:DC|AC|CR|HP)\s*\d+", re.IGNORECASE),
    ]

    # Keywords that suggest rulebook content
    RULEBOOK_KEYWORDS = {
        "spell": ["spell", "cantrip", "ritual", "incantation", "casting"],
        "rule": ["rule", "mechanic", "procedure", "resolution", "check"],
        "monster": ["monster", "creature", "beast", "enemy", "stat block"],
        "item": ["item", "weapon", "armor", "equipment", "treasure"],
        "condition": ["condition", "status", "effect", "affliction"],
        "feat": ["feat", "ability", "feature", "trait", "power"],
    }

    def __init__(self, db_manager: ChromaDBManager, similarity_threshold: float = None):
        """
        Initialize the rulebook linker.

        Args:
            db_manager: ChromaDB manager instance
        """
        self.db_manager = db_manager
        self.similarity_threshold = similarity_threshold or self.DEFAULT_SIMILARITY_THRESHOLD
        self._reference_cache: Dict[str, List[RulebookReference]] = {}

    @handle_search_errors()
    async def find_references_in_text(
        self, text: str, system: str, entity_type: str, entity_id: str, entity_name: str
    ) -> List[RulebookReference]:
        """
        Find rulebook references in a piece of text.

        Args:
            text: Text to analyze
            system: Game system (e.g., "D&D 5e")
            entity_type: Type of entity (character, npc, etc.)
            entity_id: ID of the entity
            entity_name: Name of the entity

        Returns:
            List of found references
        """
        references = []

        # Check cache first
        cache_key = f"{entity_type}:{entity_id}"
        if cache_key in self._reference_cache:
            return self._reference_cache[cache_key]

        # Find potential references using patterns
        potential_refs = self._extract_potential_references(text)

        # Search for each potential reference in rulebooks
        for ref_text, confidence in potential_refs:
            # Search in rulebook collection
            results = self.db_manager.search(
                collection_name="rulebooks",
                query=ref_text,
                n_results=3,
                metadata_filter={"system": system} if system else None,
            )

            for result in results:
                if result["distance"] < self.similarity_threshold:
                    reference = RulebookReference(
                        source_type=entity_type,
                        source_id=entity_id,
                        source_name=entity_name,
                        rulebook_id=result["id"],
                        rulebook_name=result["metadata"].get("source", "Unknown"),
                        page=result["metadata"].get("page"),
                        section=result["metadata"].get("section"),
                        content_type=result["metadata"].get("content_type", ""),
                        reference_text=ref_text,
                        confidence=confidence * (1 - result["distance"]),
                    )
                    references.append(reference)

        # Cache the results
        self._reference_cache[cache_key] = references

        return references

    def _extract_potential_references(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract potential rulebook references from text.

        Args:
            text: Text to analyze

        Returns:
            List of (reference_text, confidence) tuples
        """
        potential_refs = []

        # Check for pattern matches (patterns are already compiled)
        for pattern in self.REFERENCE_PATTERNS:
            matches = pattern.finditer(text)
            for match in matches:
                ref_text = match.group(0)
                # Higher confidence for explicit references
                confidence = (
                    0.9
                    if any(
                        keyword in ref_text.lower()
                        for keyword in ["see", "ref", "page", "PHB", "DMG"]
                    )
                    else 0.7
                )
                potential_refs.append((ref_text, confidence))

        # Check for keyword matches
        for content_type, keywords in self.RULEBOOK_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    # Extract context around keyword
                    pattern = rf"\b[^.!?]*{re.escape(keyword)}[^.!?]*[.!?]"
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        context = match.group(0).strip()
                        if len(context) > 10 and len(context) < 200:
                            potential_refs.append((context, 0.6))

        # Deduplicate while keeping highest confidence
        seen = {}
        for ref_text, confidence in potential_refs:
            key = ref_text.lower().strip()
            if key not in seen or seen[key] < confidence:
                seen[key] = confidence

        return [(text, conf) for text, conf in zip(seen.keys(), seen.values())][
            :10
        ]  # Limit to top 10

    @handle_search_errors()
    async def link_campaign_to_rulebooks(
        self, campaign: Campaign
    ) -> Dict[str, List[RulebookReference]]:
        """
        Find all rulebook references in a campaign.

        Args:
            campaign: Campaign to analyze

        Returns:
            Dictionary mapping entity IDs to their references
        """
        all_references = {}

        # Process characters
        for character in campaign.characters:
            text = f"{character.backstory} {character.notes} {character.background}"
            refs = await self.find_references_in_text(
                text, campaign.system, "character", character.id, character.name
            )
            if refs:
                all_references[f"character:{character.id}"] = refs

        # Process NPCs
        for npc in campaign.npcs:
            text = f"{npc.description} {npc.motivations} {npc.notes}"
            refs = await self.find_references_in_text(
                text, campaign.system, "npc", npc.id, npc.name
            )
            if refs:
                all_references[f"npc:{npc.id}"] = refs

        # Process locations
        for location in campaign.locations:
            text = f"{location.description} {location.notes} {' '.join(location.secrets)}"
            refs = await self.find_references_in_text(
                text, campaign.system, "location", location.id, location.name
            )
            if refs:
                all_references[f"location:{location.id}"] = refs

        # Process plot points
        for plot_point in campaign.plot_points:
            text = f"{plot_point.description} {plot_point.consequences} {plot_point.notes}"
            refs = await self.find_references_in_text(
                text, campaign.system, "plot_point", plot_point.id, plot_point.title
            )
            if refs:
                all_references[f"plot_point:{plot_point.id}"] = refs

        # Store references in database
        await self._store_references(campaign.id, all_references)

        return all_references

    async def _store_references(
        self, campaign_id: str, references: Dict[str, List[RulebookReference]]
    ) -> None:
        """
        Store references in the database.

        Args:
            campaign_id: Campaign ID
            references: References to store
        """
        try:
            # Convert references to storable format
            ref_data = {}
            for key, refs in references.items():
                ref_data[key] = [ref.to_dict() for ref in refs]

            # Store in campaigns collection
            self.db_manager.add_document(
                collection_name="campaigns",
                document_id=f"{campaign_id}_references",
                content=json.dumps(ref_data),
                metadata={
                    "type": "campaign_references",
                    "campaign_id": campaign_id,
                    "reference_count": sum(len(refs) for refs in references.values()),
                },
            )

            logger.info(f"Stored {len(ref_data)} reference groups for campaign {campaign_id}")

        except Exception as e:
            logger.error(f"Failed to store references: {str(e)}")

    @handle_search_errors()
    async def get_references_for_entity(
        self, campaign_id: str, entity_type: str, entity_id: str
    ) -> List[RulebookReference]:
        """
        Get rulebook references for a specific entity.

        Args:
            campaign_id: Campaign ID
            entity_type: Type of entity
            entity_id: Entity ID

        Returns:
            List of references
        """
        try:
            # Try cache first
            cache_key = f"{entity_type}:{entity_id}"
            if cache_key in self._reference_cache:
                return self._reference_cache[cache_key]

            # Get from database
            doc = self.db_manager.get_document("campaigns", f"{campaign_id}_references")

            if doc:
                ref_data = json.loads(doc["content"])
                if cache_key in ref_data:
                    refs = [RulebookReference(**ref_dict) for ref_dict in ref_data[cache_key]]
                    self._reference_cache[cache_key] = refs
                    return refs

            return []

        except Exception as e:
            logger.error(f"Failed to get references: {str(e)}")
            return []

    @handle_search_errors()
    async def validate_references(self, campaign_id: str) -> Dict[str, Any]:
        """
        Validate all references in a campaign to check for broken links.

        Args:
            campaign_id: Campaign ID

        Returns:
            Validation report
        """
        try:
            # Get all references
            doc = self.db_manager.get_document("campaigns", f"{campaign_id}_references")

            if not doc:
                return {"valid": True, "broken_references": [], "message": "No references found"}

            ref_data = json.loads(doc["content"])
            broken_refs = []

            # Check each reference
            for entity_key, refs in ref_data.items():
                for ref_dict in refs:
                    ref = RulebookReference(**ref_dict)

                    # Check if the rulebook document still exists
                    rulebook_doc = self.db_manager.get_document("rulebooks", ref.rulebook_id)

                    if not rulebook_doc:
                        broken_refs.append(
                            {
                                "entity": entity_key,
                                "reference": ref.reference_text,
                                "rulebook": ref.rulebook_name,
                                "reason": "Rulebook document not found",
                            }
                        )

            return {
                "valid": len(broken_refs) == 0,
                "broken_references": broken_refs,
                "total_references": sum(len(refs) for refs in ref_data.values()),
                "broken_count": len(broken_refs),
            }

        except Exception as e:
            logger.error(f"Failed to validate references: {str(e)}")
            return {"valid": False, "error": str(e)}

    @handle_search_errors()
    async def get_campaign_context_for_search(self, campaign_id: str, query: str) -> Dict[str, Any]:
        """
        Get campaign context relevant to a search query.

        Args:
            campaign_id: Campaign ID
            query: Search query

        Returns:
            Campaign context data
        """
        try:
            # Search campaign data for relevant context
            campaign_results = self.db_manager.search(
                collection_name="campaigns",
                query=query,
                n_results=5,
                metadata_filter={"campaign_id": campaign_id},
            )

            context = {"campaign_entities": [], "related_references": []}

            # Process campaign results
            for result in campaign_results:
                if result["metadata"].get("type") == "campaign":
                    # Extract relevant entities from campaign data
                    campaign_data = json.loads(result["content"])
                    # Add logic to extract relevant entities based on query

            # Get related rulebook references
            if campaign_results:
                doc = self.db_manager.get_document("campaigns", f"{campaign_id}_references")

                if doc:
                    ref_data = json.loads(doc["content"])
                    # Find references related to the query
                    for entity_key, refs in ref_data.items():
                        for ref_dict in refs:
                            ref = RulebookReference(**ref_dict)
                            if query.lower() in ref.reference_text.lower():
                                context["related_references"].append(ref.to_dict())

            return context

        except Exception as e:
            logger.error(f"Failed to get campaign context: {str(e)}")
            return {"campaign_entities": [], "related_references": []}
