"""Source organization and categorization system."""

import logging
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict

from .models import (
    Source,
    SourceCategory,
    SourceRelationship,
    ContentCategory,
    SourceType,
    SourceMetadata
)

logger = logging.getLogger(__name__)


class SourceOrganizer:
    """Organize and categorize sources for efficient retrieval."""
    
    # Keywords for automatic categorization
    CATEGORY_KEYWORDS = {
        ContentCategory.RULES: [
            'rules', 'mechanics', 'system', 'gameplay', 'resolution',
            'dice', 'check', 'modifier', 'difficulty', 'action'
        ],
        ContentCategory.MECHANICS: [
            'combat', 'initiative', 'attack', 'defense', 'damage',
            'hit points', 'armor class', 'saving throw', 'skill check'
        ],
        ContentCategory.LORE: [
            'history', 'legend', 'mythology', 'ancient', 'prophecy',
            'timeline', 'era', 'age', 'civilization', 'culture'
        ],
        ContentCategory.NARRATIVE: [
            'story', 'plot', 'adventure', 'quest', 'campaign',
            'scene', 'encounter', 'event', 'description'
        ],
        ContentCategory.CHARACTERS: [
            'character', 'npc', 'personality', 'background', 'motivation',
            'ally', 'enemy', 'faction', 'organization'
        ],
        ContentCategory.LOCATIONS: [
            'location', 'place', 'city', 'town', 'dungeon',
            'realm', 'plane', 'world', 'map', 'geography'
        ],
        ContentCategory.ITEMS: [
            'item', 'equipment', 'weapon', 'armor', 'gear',
            'treasure', 'artifact', 'magic item', 'potion'
        ],
        ContentCategory.SPELLS: [
            'spell', 'magic', 'cantrip', 'ritual', 'incantation',
            'school of magic', 'arcane', 'divine', 'component'
        ],
        ContentCategory.MONSTERS: [
            'monster', 'creature', 'beast', 'enemy', 'stat block',
            'challenge rating', 'legendary', 'lair', 'minion'
        ],
        ContentCategory.CLASSES: [
            'class', 'profession', 'archetype', 'subclass', 'specialization',
            'level', 'feature', 'ability', 'progression'
        ],
        ContentCategory.RACES: [
            'race', 'species', 'ancestry', 'lineage', 'heritage',
            'trait', 'racial', 'subrace', 'variant'
        ],
        ContentCategory.FEATS: [
            'feat', 'talent', 'perk', 'advantage', 'special ability',
            'prerequisite', 'benefit', 'enhancement'
        ],
        ContentCategory.SKILLS: [
            'skill', 'proficiency', 'expertise', 'training', 'competence',
            'check', 'ability score', 'modifier'
        ],
        ContentCategory.TABLES: [
            'table', 'chart', 'random', 'roll', 'result',
            'd20', 'd100', 'generator', 'list'
        ]
    }
    
    # Relationship patterns
    RELATIONSHIP_PATTERNS = {
        'supplement_to': [
            r'supplement to (.*)',
            r'expands (.*)',
            r'companion to (.*)',
            r'adds to (.*)'
        ],
        'requires': [
            r'requires (.*)',
            r'needs (.*)',
            r'must have (.*)',
            r'prerequisite: (.*)'
        ],
        'replaces': [
            r'replaces (.*)',
            r'supersedes (.*)',
            r'updates (.*)',
            r'revision of (.*)'
        ],
        'conflicts_with': [
            r'incompatible with (.*)',
            r'conflicts with (.*)',
            r'cannot be used with (.*)'
        ]
    }
    
    def __init__(self):
        """Initialize the source organizer."""
        self.category_cache = {}
        self.relationship_cache = defaultdict(list)
    
    def categorize_source(
        self,
        source: Source,
        content_sample: Optional[str] = None,
        auto_categorize: bool = True
    ) -> List[SourceCategory]:
        """
        Categorize a source based on its content.
        
        Args:
            source: Source to categorize
            content_sample: Optional content sample for analysis
            auto_categorize: Whether to auto-detect categories
            
        Returns:
            List of assigned categories
        """
        categories = []
        
        # Start with manual categories from metadata
        if source.metadata.categories:
            for cat in source.metadata.categories:
                category = SourceCategory(
                    name=cat.value if hasattr(cat, 'value') else str(cat),
                    category_type=cat if isinstance(cat, ContentCategory) else ContentCategory.RULES,
                    auto_assigned=False,
                    is_primary=True
                )
                categories.append(category)
        
        # Auto-categorize if enabled
        if auto_categorize:
            # Use content sample or extract from chunks
            if not content_sample and source.content_chunks:
                # Sample first few chunks
                sample_chunks = source.content_chunks[:5]
                content_sample = ' '.join([
                    chunk.get('content', '')[:500]
                    for chunk in sample_chunks
                ])
            
            if content_sample:
                auto_categories = self._detect_categories(content_sample)
                
                # Add auto-detected categories
                existing_types = {cat.category_type for cat in categories}
                for cat_type, score in auto_categories:
                    if cat_type not in existing_types and score > 0.3:
                        category = SourceCategory(
                            name=cat_type.value,
                            category_type=cat_type,
                            auto_assigned=True,
                            is_primary=score > 0.5,
                            priority=int(score * 100)
                        )
                        categories.append(category)
                        category.source_ids.append(source.id)
        
        # Organize by hierarchy
        categories = self._organize_hierarchy(categories)
        
        # Cache for quick lookup
        self.category_cache[source.id] = categories
        
        return categories
    
    def detect_relationships(
        self,
        source: Source,
        existing_sources: List[Source]
    ) -> List[SourceRelationship]:
        """
        Detect relationships between sources.
        
        Args:
            source: Source to analyze
            existing_sources: List of existing sources to check against
            
        Returns:
            List of detected relationships
        """
        relationships = []
        
        # Check metadata for explicit relationships
        description = source.metadata.description or ''
        title = source.metadata.title
        
        for rel_type, patterns in self.RELATIONSHIP_PATTERNS.items():
            for pattern in patterns:
                # Check in description
                match = re.search(pattern, description, re.IGNORECASE)
                if match:
                    related_name = match.group(1).strip()
                    related_source = self._find_source_by_name(
                        related_name, existing_sources
                    )
                    
                    if related_source:
                        relationship = SourceRelationship(
                            source_id=source.id,
                            related_source_id=related_source.id,
                            relationship_type=rel_type,
                            description=f"Detected from metadata: {match.group(0)}",
                            strength=0.8
                        )
                        relationships.append(relationship)
        
        # Check for series relationships
        series_rel = self._detect_series_relationship(source, existing_sources)
        if series_rel:
            relationships.append(series_rel)
        
        # Check for system relationships
        system_rels = self._detect_system_relationships(source, existing_sources)
        relationships.extend(system_rels)
        
        # Cache relationships
        self.relationship_cache[source.id] = relationships
        
        return relationships
    
    def organize_by_system(
        self,
        sources: List[Source]
    ) -> Dict[str, List[Source]]:
        """
        Organize sources by game system.
        
        Args:
            sources: List of sources to organize
            
        Returns:
            Dictionary mapping system names to source lists
        """
        organized = defaultdict(list)
        
        for source in sources:
            system = source.metadata.system
            organized[system].append(source)
        
        # Sort sources within each system
        for system in organized:
            organized[system].sort(
                key=lambda s: (
                    s.metadata.source_type.value,
                    s.metadata.title
                )
            )
        
        return dict(organized)
    
    def organize_by_type(
        self,
        sources: List[Source]
    ) -> Dict[SourceType, List[Source]]:
        """
        Organize sources by type.
        
        Args:
            sources: List of sources to organize
            
        Returns:
            Dictionary mapping source types to source lists
        """
        organized = defaultdict(list)
        
        for source in sources:
            source_type = source.metadata.source_type
            organized[source_type].append(source)
        
        return dict(organized)
    
    def organize_by_category(
        self,
        sources: List[Source]
    ) -> Dict[ContentCategory, List[Source]]:
        """
        Organize sources by content category.
        
        Args:
            sources: List of sources to organize
            
        Returns:
            Dictionary mapping categories to source lists
        """
        organized = defaultdict(list)
        
        for source in sources:
            # Get categories from cache or source
            categories = self.category_cache.get(source.id, source.categories)
            
            for category in categories:
                cat_type = category.category_type
                organized[cat_type].append(source)
        
        return dict(organized)
    
    def find_related_sources(
        self,
        source_id: str,
        relationship_type: Optional[str] = None,
        max_depth: int = 1
    ) -> List[Tuple[Source, SourceRelationship]]:
        """
        Find sources related to a given source.
        
        Args:
            source_id: ID of the source
            relationship_type: Optional specific relationship type
            max_depth: Maximum depth for traversal
            
        Returns:
            List of (source, relationship) tuples
        """
        related = []
        visited = set()
        
        def traverse(sid: str, depth: int):
            if depth > max_depth or sid in visited:
                return
            
            visited.add(sid)
            
            # Get relationships from cache
            relationships = self.relationship_cache.get(sid, [])
            
            for rel in relationships:
                if relationship_type and rel.relationship_type != relationship_type:
                    continue
                
                # Add related source
                related.append((rel.related_source_id, rel))
                
                # Traverse if not at max depth
                if depth < max_depth:
                    traverse(rel.related_source_id, depth + 1)
        
        traverse(source_id, 0)
        
        return related
    
    def build_dependency_tree(
        self,
        source: Source,
        existing_sources: List[Source]
    ) -> Dict[str, Any]:
        """
        Build a dependency tree for a source.
        
        Args:
            source: Source to analyze
            existing_sources: List of existing sources
            
        Returns:
            Dependency tree structure
        """
        tree = {
            "source_id": source.id,
            "title": source.metadata.title,
            "requires": [],
            "supplements": [],
            "conflicts": []
        }
        
        relationships = self.relationship_cache.get(source.id, [])
        
        for rel in relationships:
            related = self._find_source_by_id(rel.related_source_id, existing_sources)
            if not related:
                continue
            
            rel_info = {
                "id": related.id,
                "title": related.metadata.title,
                "strength": rel.strength
            }
            
            if rel.relationship_type == "requires":
                tree["requires"].append(rel_info)
            elif rel.relationship_type in ["supplement_to", "expands"]:
                tree["supplements"].append(rel_info)
            elif rel.relationship_type == "conflicts_with":
                tree["conflicts"].append(rel_info)
        
        return tree
    
    def _detect_categories(self, content: str) -> List[Tuple[ContentCategory, float]]:
        """Detect categories from content using keyword matching."""
        content_lower = content.lower()
        category_scores = []
        
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            score = 0
            keyword_count = 0
            
            for keyword in keywords:
                if keyword in content_lower:
                    keyword_count += 1
                    # Weight by keyword specificity
                    score += (1.0 / len(keywords))
            
            if keyword_count > 0:
                # Normalize score
                score = min(1.0, score * (keyword_count / 3))
                category_scores.append((category, score))
        
        # Sort by score
        category_scores.sort(key=lambda x: x[1], reverse=True)
        
        return category_scores
    
    def _organize_hierarchy(
        self,
        categories: List[SourceCategory]
    ) -> List[SourceCategory]:
        """Organize categories into a hierarchy."""
        # Define parent-child relationships
        hierarchy = {
            ContentCategory.MECHANICS: [ContentCategory.COMBAT],
            ContentCategory.RULES: [ContentCategory.MECHANICS, ContentCategory.SKILLS],
            ContentCategory.NARRATIVE: [ContentCategory.ADVENTURES],
            ContentCategory.LORE: [ContentCategory.LOCATIONS, ContentCategory.CHARACTERS]
        }
        
        # Set parent categories
        for category in categories:
            for parent, children in hierarchy.items():
                if category.category_type in children:
                    # Find parent category
                    parent_cat = next(
                        (c for c in categories if c.category_type == parent),
                        None
                    )
                    if parent_cat:
                        category.parent_category = parent_cat.name
                        parent_cat.subcategories.append(category.name)
        
        return categories
    
    def _detect_series_relationship(
        self,
        source: Source,
        existing_sources: List[Source]
    ) -> Optional[SourceRelationship]:
        """Detect if source is part of a series."""
        title = source.metadata.title
        
        # Look for volume/part indicators
        series_patterns = [
            r'(.+?)\s+(?:Vol|Volume|Part|Book)\s+(\d+)',
            r'(.+?)\s+(\d+)$',
            r'(.+?):\s+(.+?)$'  # Title: Subtitle pattern
        ]
        
        for pattern in series_patterns:
            match = re.match(pattern, title)
            if match:
                base_title = match.group(1).strip()
                
                # Find related sources with similar titles
                for other in existing_sources:
                    if other.id == source.id:
                        continue
                    
                    if base_title.lower() in other.metadata.title.lower():
                        return SourceRelationship(
                            source_id=source.id,
                            related_source_id=other.id,
                            relationship_type="part_of_series",
                            description=f"Part of '{base_title}' series",
                            strength=0.7,
                            bidirectional=True
                        )
        
        return None
    
    def _detect_system_relationships(
        self,
        source: Source,
        existing_sources: List[Source]
    ) -> List[SourceRelationship]:
        """Detect relationships based on game system."""
        relationships = []
        
        # Find sources from the same system
        same_system = [
            s for s in existing_sources
            if s.metadata.system == source.metadata.system and s.id != source.id
        ]
        
        # Core rulebook relationship
        if source.metadata.source_type != SourceType.RULEBOOK:
            for other in same_system:
                if other.metadata.source_type == SourceType.RULEBOOK:
                    # Check if it's the core rulebook
                    if 'core' in other.metadata.title.lower() or 'player' in other.metadata.title.lower():
                        relationships.append(SourceRelationship(
                            source_id=source.id,
                            related_source_id=other.id,
                            relationship_type="requires",
                            description="Requires core rulebook",
                            strength=0.9
                        ))
                        break
        
        return relationships
    
    def _find_source_by_name(
        self,
        name: str,
        sources: List[Source]
    ) -> Optional[Source]:
        """Find a source by name (fuzzy matching)."""
        name_lower = name.lower()
        
        # Exact match
        for source in sources:
            if source.metadata.title.lower() == name_lower:
                return source
        
        # Partial match
        for source in sources:
            if name_lower in source.metadata.title.lower():
                return source
        
        # Very fuzzy match (first few words)
        name_words = name_lower.split()[:3]
        for source in sources:
            title_words = source.metadata.title.lower().split()[:3]
            if name_words == title_words:
                return source
        
        return None
    
    def _find_source_by_id(
        self,
        source_id: str,
        sources: List[Source]
    ) -> Optional[Source]:
        """Find a source by ID."""
        for source in sources:
            if source.id == source_id:
                return source
        return None