"""Flavor source integration and narrative extraction system."""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import random

from .models import FlavorSource, Source, SourceType

logger = logging.getLogger(__name__)


class FlavorIntegrator:
    """Integrate flavor sources for narrative generation."""
    
    # Narrative element patterns
    NARRATIVE_PATTERNS = {
        'character': [
            r'(?:named?|called?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:was|is|became)',
            r'(?:the\s+)?([A-Z][a-z]+(?:\s+(?:the\s+)?[A-Z][a-z]+)*),?\s+(?:a|an|the)\s+\w+'
        ],
        'location': [
            r'(?:in|at|near|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:city|town|village|kingdom|realm)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Mountains|Forest|Desert|Plains|Sea|Ocean|River)'
        ],
        'event': [
            r'(?:Battle|War|Conquest|Fall)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?:The\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Rebellion|Revolution|Uprising)',
            r'(?:when|after|before)\s+the\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        ],
        'quote': [
            r'"([^"]+)"',
            r"'([^']+)'",
            r'[""]([^""]+)[""]'
        ]
    }
    
    # Style indicators
    STYLE_INDICATORS = {
        'epic': ['mighty', 'legendary', 'heroic', 'glorious', 'destiny'],
        'dark': ['shadow', 'darkness', 'corruption', 'doom', 'despair'],
        'mystical': ['arcane', 'mystical', 'ethereal', 'otherworldly', 'enchanted'],
        'gritty': ['blood', 'steel', 'survival', 'harsh', 'brutal'],
        'whimsical': ['curious', 'peculiar', 'delightful', 'charming', 'merry'],
        'horror': ['terror', 'nightmare', 'dread', 'horror', 'unspeakable']
    }
    
    # Tone indicators
    TONE_INDICATORS = {
        'serious': ['grave', 'solemn', 'critical', 'dire', 'urgent'],
        'humorous': ['laughed', 'jest', 'amusing', 'ridiculous', 'absurd'],
        'mysterious': ['unknown', 'hidden', 'secret', 'enigmatic', 'cryptic'],
        'adventurous': ['quest', 'journey', 'explore', 'discover', 'venture'],
        'tragic': ['sorrow', 'loss', 'grief', 'tragedy', 'mourning']
    }
    
    def __init__(self):
        """Initialize the flavor integrator."""
        self.flavor_cache = {}
        self.narrative_elements = defaultdict(list)
    
    def process_flavor_source(
        self,
        source: Source,
        content_chunks: Optional[List[Dict[str, Any]]] = None
    ) -> FlavorSource:
        """
        Process a source as flavor material.
        
        Args:
            source: Source to process
            content_chunks: Optional content chunks to analyze
            
        Returns:
            FlavorSource with extracted narrative elements
        """
        # Convert to FlavorSource if needed
        if isinstance(source, FlavorSource):
            flavor = source
        else:
            flavor = FlavorSource()
            flavor.__dict__.update(source.__dict__)
        
        chunks = content_chunks or source.content_chunks
        
        if not chunks:
            logger.warning(f"No content chunks for flavor source {source.id}")
            return flavor
        
        # Extract narrative elements
        elements = self._extract_narrative_elements(chunks)
        
        flavor.characters = elements.get('characters', [])
        flavor.locations = elements.get('locations', [])
        flavor.events = elements.get('events', [])
        flavor.quotes = elements.get('quotes', [])
        
        # Detect style and tone
        full_text = ' '.join([chunk.get('content', '')[:1000] for chunk in chunks[:10]])
        
        flavor.narrative_style = self._detect_style(full_text)
        flavor.tone = self._detect_tone(full_text)
        
        # Extract themes
        flavor.themes = self._extract_themes(full_text)
        
        # Set creativity modifier based on source type
        if source.metadata.source_type == SourceType.FLAVOR:
            flavor.creativity_modifier = 1.2
        elif source.metadata.source_type == SourceType.ADVENTURE:
            flavor.creativity_modifier = 1.1
        else:
            flavor.creativity_modifier = 0.8
        
        # Cache processed flavor
        self.flavor_cache[flavor.id] = flavor
        
        return flavor
    
    def blend_flavor_sources(
        self,
        sources: List[FlavorSource],
        weights: Optional[Dict[str, float]] = None,
        conflict_resolution: str = "priority"  # "priority", "blend", "random"
    ) -> Dict[str, Any]:
        """
        Blend multiple flavor sources intelligently.
        
        Args:
            sources: List of flavor sources to blend
            weights: Optional weights for each source
            conflict_resolution: How to handle conflicts
            
        Returns:
            Blended narrative elements
        """
        if not sources:
            return {}
        
        blended = {
            'narrative_style': '',
            'tone': '',
            'themes': [],
            'characters': [],
            'locations': [],
            'events': [],
            'quotes': [],
            'creativity_modifier': 1.0
        }
        
        # Sort by priority if using priority resolution
        if conflict_resolution == "priority":
            sources = sorted(sources, key=lambda s: s.priority, reverse=True)
        
        # Blend narrative styles
        styles = [s.narrative_style for s in sources if s.narrative_style]
        if styles:
            if conflict_resolution == "blend":
                blended['narrative_style'] = self._blend_styles(styles)
            elif conflict_resolution == "random":
                blended['narrative_style'] = random.choice(styles)
            else:  # priority
                blended['narrative_style'] = styles[0]
        
        # Blend tones
        tones = [s.tone for s in sources if s.tone]
        if tones:
            if conflict_resolution == "blend":
                blended['tone'] = self._blend_tones(tones)
            else:
                blended['tone'] = tones[0] if conflict_resolution == "priority" else random.choice(tones)
        
        # Merge themes (unique)
        all_themes = set()
        for source in sources:
            all_themes.update(source.themes)
        blended['themes'] = list(all_themes)
        
        # Merge narrative elements with weighting
        for element_type in ['characters', 'locations', 'events', 'quotes']:
            merged = self._merge_narrative_elements(
                sources,
                element_type,
                weights,
                conflict_resolution
            )
            blended[element_type] = merged
        
        # Calculate blended creativity modifier
        if weights:
            total_weight = sum(weights.values())
            blended['creativity_modifier'] = sum(
                source.creativity_modifier * weights.get(source.id, 1.0)
                for source in sources
            ) / total_weight if total_weight > 0 else 1.0
        else:
            blended['creativity_modifier'] = sum(
                s.creativity_modifier for s in sources
            ) / len(sources)
        
        return blended
    
    def apply_flavor_to_content(
        self,
        content: str,
        flavor_source: FlavorSource,
        intensity: float = 1.0  # 0.0 to 2.0
    ) -> str:
        """
        Apply flavor source style to content.
        
        Args:
            content: Content to transform
            flavor_source: Flavor source to apply
            intensity: How strongly to apply the flavor
            
        Returns:
            Transformed content
        """
        transformed = content
        
        # Apply style transformations
        if flavor_source.narrative_style:
            transformed = self._apply_style(
                transformed,
                flavor_source.narrative_style,
                intensity
            )
        
        # Apply tone
        if flavor_source.tone:
            transformed = self._apply_tone(
                transformed,
                flavor_source.tone,
                intensity
            )
        
        # Inject narrative elements if appropriate
        if intensity > 0.5:
            transformed = self._inject_narrative_elements(
                transformed,
                flavor_source,
                min(intensity, 1.0)
            )
        
        return transformed
    
    def generate_flavor_context(
        self,
        query: str,
        flavor_sources: List[FlavorSource],
        max_elements: int = 5
    ) -> Dict[str, Any]:
        """
        Generate contextual flavor information for a query.
        
        Args:
            query: Query to contextualize
            flavor_sources: Available flavor sources
            max_elements: Maximum elements to include
            
        Returns:
            Contextual flavor information
        """
        context = {
            'relevant_characters': [],
            'relevant_locations': [],
            'relevant_events': [],
            'relevant_quotes': [],
            'suggested_style': '',
            'suggested_tone': ''
        }
        
        query_lower = query.lower()
        
        # Find relevant elements
        for source in flavor_sources:
            # Check characters
            for char in source.characters[:max_elements]:
                if any(word in query_lower for word in char.get('name', '').lower().split()):
                    context['relevant_characters'].append(char)
            
            # Check locations
            for loc in source.locations[:max_elements]:
                if any(word in query_lower for word in loc.get('name', '').lower().split()):
                    context['relevant_locations'].append(loc)
            
            # Check events
            for event in source.events[:max_elements]:
                if any(word in query_lower for word in event.get('name', '').lower().split()):
                    context['relevant_events'].append(event)
        
        # Limit results
        context['relevant_characters'] = context['relevant_characters'][:max_elements]
        context['relevant_locations'] = context['relevant_locations'][:max_elements]
        context['relevant_events'] = context['relevant_events'][:max_elements]
        
        # Add relevant quotes
        if flavor_sources:
            relevant_quotes = []
            for source in flavor_sources:
                for quote in source.quotes:
                    if any(word in query_lower for word in quote.get('text', '').lower().split()):
                        relevant_quotes.append(quote)
            
            context['relevant_quotes'] = relevant_quotes[:max_elements]
        
        # Suggest style and tone based on query
        if 'battle' in query_lower or 'combat' in query_lower:
            context['suggested_style'] = 'epic'
            context['suggested_tone'] = 'serious'
        elif 'mystery' in query_lower or 'secret' in query_lower:
            context['suggested_style'] = 'mystical'
            context['suggested_tone'] = 'mysterious'
        elif 'adventure' in query_lower or 'quest' in query_lower:
            context['suggested_style'] = 'epic'
            context['suggested_tone'] = 'adventurous'
        
        return context
    
    def _extract_narrative_elements(
        self,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Extract narrative elements from content chunks."""
        elements = {
            'characters': [],
            'locations': [],
            'events': [],
            'quotes': []
        }
        
        seen = {
            'characters': set(),
            'locations': set(),
            'events': set(),
            'quotes': set()
        }
        
        for chunk in chunks:
            content = chunk.get('content', '')
            
            # Extract each type of element
            for element_type, patterns in self.NARRATIVE_PATTERNS.items():
                element_key = element_type + 's'  # pluralize
                
                for pattern in patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        extracted = match.group(1) if match.groups() else match.group(0)
                        
                        # Avoid duplicates
                        if extracted not in seen[element_key]:
                            seen[element_key].add(extracted)
                            
                            element_dict = {
                                'name' if element_type != 'quote' else 'text': extracted,
                                'source_chunk': chunk.get('id', ''),
                                'context': content[max(0, match.start()-50):min(len(content), match.end()+50)]
                            }
                            
                            elements[element_key].append(element_dict)
        
        return elements
    
    def _detect_style(self, text: str) -> str:
        """Detect narrative style from text."""
        text_lower = text.lower()
        style_scores = {}
        
        for style, indicators in self.STYLE_INDICATORS.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            if score > 0:
                style_scores[style] = score
        
        if not style_scores:
            return 'neutral'
        
        # Return the style with highest score
        return max(style_scores, key=style_scores.get)
    
    def _detect_tone(self, text: str) -> str:
        """Detect tone from text."""
        text_lower = text.lower()
        tone_scores = {}
        
        for tone, indicators in self.TONE_INDICATORS.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            if score > 0:
                tone_scores[tone] = score
        
        if not tone_scores:
            return 'neutral'
        
        return max(tone_scores, key=tone_scores.get)
    
    def _extract_themes(self, text: str) -> List[str]:
        """Extract themes from text."""
        themes = []
        text_lower = text.lower()
        
        # Common fantasy themes
        theme_keywords = {
            'redemption': ['redeem', 'forgive', 'atone', 'salvation'],
            'corruption': ['corrupt', 'decay', 'taint', 'fall'],
            'sacrifice': ['sacrifice', 'gave up', 'cost', 'price'],
            'destiny': ['destiny', 'fate', 'prophecy', 'foretold'],
            'power': ['power', 'strength', 'dominate', 'control'],
            'honor': ['honor', 'duty', 'loyalty', 'oath'],
            'revenge': ['revenge', 'vengeance', 'retribution', 'payback'],
            'love': ['love', 'passion', 'romance', 'devotion'],
            'war': ['war', 'battle', 'conflict', 'siege'],
            'discovery': ['discover', 'explore', 'uncover', 'reveal']
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                themes.append(theme)
        
        return themes
    
    def _blend_styles(self, styles: List[str]) -> str:
        """Blend multiple styles together."""
        if not styles:
            return 'neutral'
        
        # Count occurrences
        style_counts = defaultdict(int)
        for style in styles:
            style_counts[style] += 1
        
        # If one style dominates, use it
        max_count = max(style_counts.values())
        if max_count > len(styles) / 2:
            return max(style_counts, key=style_counts.get)
        
        # Otherwise, create a compound style
        top_styles = sorted(style_counts.keys(), key=style_counts.get, reverse=True)[:2]
        return f"{top_styles[0]}-{top_styles[1]}"
    
    def _blend_tones(self, tones: List[str]) -> str:
        """Blend multiple tones together."""
        if not tones:
            return 'neutral'
        
        # Similar to style blending
        tone_counts = defaultdict(int)
        for tone in tones:
            tone_counts[tone] += 1
        
        max_count = max(tone_counts.values())
        if max_count > len(tones) / 2:
            return max(tone_counts, key=tone_counts.get)
        
        top_tones = sorted(tone_counts.keys(), key=tone_counts.get, reverse=True)[:2]
        return f"{top_tones[0]} with hints of {top_tones[1]}"
    
    def _merge_narrative_elements(
        self,
        sources: List[FlavorSource],
        element_type: str,
        weights: Optional[Dict[str, float]],
        conflict_resolution: str
    ) -> List[Dict[str, Any]]:
        """Merge narrative elements from multiple sources."""
        merged = []
        seen = set()
        
        for source in sources:
            weight = weights.get(source.id, 1.0) if weights else 1.0
            elements = getattr(source, element_type, [])
            
            for element in elements:
                # Use name or text as identifier
                identifier = element.get('name') or element.get('text', '')
                
                if identifier not in seen:
                    seen.add(identifier)
                    # Add weight information
                    element['weight'] = weight
                    element['source_id'] = source.id
                    merged.append(element)
        
        # Sort by weight if using priority
        if conflict_resolution == "priority":
            merged.sort(key=lambda x: x.get('weight', 0), reverse=True)
        
        return merged
    
    def _apply_style(self, content: str, style: str, intensity: float) -> str:
        """Apply a narrative style to content."""
        if intensity < 0.1:
            return content
        
        # Style-specific transformations
        if style == 'epic' and intensity > 0.5:
            # Add epic descriptors
            epic_words = ['mighty', 'legendary', 'glorious']
            # Simple example - in practice, would use NLP
            # Use regex to only replace standalone 'warrior'
            def epic_warrior_repl(match):
                return f"{random.choice(epic_words)} warrior"
            content = re.sub(r'\bwarrior\b', epic_warrior_repl, content)
            content = content.replace('battle', 'epic battle')
        
        elif style == 'dark' and intensity > 0.5:
            # Add dark atmosphere
            dark_words = ['shadowy', 'grim', 'foreboding']
            content = content.replace('forest', f'{random.choice(dark_words)} forest')
            content = content.replace('path', 'darkened path')
        
        # More styles could be added...
        
        return content
    
    def _apply_tone(self, content: str, tone: str, intensity: float) -> str:
        """Apply a tone to content."""
        if intensity < 0.1:
            return content
        
        # Tone-specific adjustments
        if tone == 'mysterious' and intensity > 0.5:
            # Add mysterious elements
            content = content.replace('. ', '... ')
            
        elif tone == 'adventurous' and intensity > 0.5:
            # Add excitement
            content = content[:len(content)//3].replace('.', '!') + content[len(content)//3:]
        
        return content
    
    def _inject_narrative_elements(
        self,
        content: str,
        flavor_source: FlavorSource,
        probability: float
    ) -> str:
        """Inject narrative elements into content."""
        if random.random() > probability:
            return content
        
        # Randomly inject a character or location reference
        if flavor_source.characters and random.random() < 0.5:
            char = random.choice(flavor_source.characters[:3])
            content += f" (as {char.get('name', 'someone')} once said)"
        
        elif flavor_source.locations and random.random() < 0.5:
            loc = random.choice(flavor_source.locations[:3])
            content += f" (like in {loc.get('name', 'that place')})"
        
        return content