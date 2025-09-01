"""Ebook extraction module for enriching TTRPG character generation content.

This module processes EPUB and MOBI files to extract useful TTRPG elements
such as character traits, backgrounds, NPC archetypes, story hooks, and
world-building details.
"""

import json
import logging
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from xml.etree import ElementTree as ET

import ebooklib
import mobi
from bs4 import BeautifulSoup
from ebooklib import epub

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExtractedContent:
    """Container for extracted TTRPG content from ebooks."""
    
    # Character elements
    character_traits: Set[str] = field(default_factory=set)
    backgrounds: Set[str] = field(default_factory=set)
    motivations: Set[str] = field(default_factory=set)
    flaws: Set[str] = field(default_factory=set)
    ideals: Set[str] = field(default_factory=set)
    bonds: Set[str] = field(default_factory=set)
    fears: Set[str] = field(default_factory=set)
    
    # NPC elements
    npc_archetypes: Set[str] = field(default_factory=set)
    npc_personalities: Set[str] = field(default_factory=set)
    npc_occupations: Set[str] = field(default_factory=set)
    npc_quirks: Set[str] = field(default_factory=set)
    
    # Story elements
    story_hooks: Set[str] = field(default_factory=set)
    plot_twists: Set[str] = field(default_factory=set)
    quest_ideas: Set[str] = field(default_factory=set)
    conflicts: Set[str] = field(default_factory=set)
    
    # World-building
    locations: Set[str] = field(default_factory=set)
    factions: Set[str] = field(default_factory=set)
    cultural_elements: Set[str] = field(default_factory=set)
    technologies: Set[str] = field(default_factory=set)
    magic_systems: Set[str] = field(default_factory=set)
    
    # Names and titles
    character_names: Set[str] = field(default_factory=set)
    place_names: Set[str] = field(default_factory=set)
    organization_names: Set[str] = field(default_factory=set)
    titles_and_ranks: Set[str] = field(default_factory=set)
    
    # Items and equipment
    weapons: Set[str] = field(default_factory=set)
    armor: Set[str] = field(default_factory=set)
    items: Set[str] = field(default_factory=set)
    vehicles: Set[str] = field(default_factory=set)
    
    def merge(self, other: 'ExtractedContent') -> None:
        """Merge another ExtractedContent instance into this one."""
        for field_name in self.__dataclass_fields__:
            current_set = getattr(self, field_name)
            other_set = getattr(other, field_name)
            current_set.update(other_set)
    
    def to_dict(self) -> Dict[str, List[str]]:
        """Convert to dictionary with lists instead of sets."""
        return {
            field_name: sorted(list(getattr(self, field_name)))
            for field_name in self.__dataclass_fields__
        }
    
    def filter_and_clean(self) -> None:
        """Filter and clean extracted content."""
        for field_name in self.__dataclass_fields__:
            current_set = getattr(self, field_name)
            # Remove empty strings and very short entries
            cleaned = {
                item.strip() for item in current_set 
                if item and len(item.strip()) > 2
            }
            # Remove duplicates with different capitalization
            unique_items = {}
            for item in cleaned:
                lower_key = item.lower()
                if lower_key not in unique_items or len(item) > len(unique_items[lower_key]):
                    unique_items[lower_key] = item
            setattr(self, field_name, set(unique_items.values()))


class ContentPatternMatcher:
    """Pattern matching for extracting TTRPG content from text."""
    
    # Invalid traits to filter out
    _INVALID_TRAITS = {
        'very', 'quite', 'rather', 'somewhat', 'really', 'truly',
        'more', 'less', 'most', 'least', 'much', 'many', 'few',
        'good', 'bad', 'nice', 'okay', 'fine', 'great'
    }
    
    # Invalid occupations to filter out
    _INVALID_OCCUPATIONS = {
        'person', 'man', 'woman', 'child', 'boy', 'girl',
        'one', 'someone', 'somebody', 'anyone', 'anybody',
        'thing', 'stuff', 'matter', 'issue', 'problem'
    }
    
    # Invalid locations to filter out
    _INVALID_LOCATIONS = {
        'place', 'area', 'spot', 'location', 'site',
        'here', 'there', 'where', 'somewhere', 'anywhere',
        'thing', 'stuff', 'matter', 'way', 'direction'
    }
    
    def __init__(self):
        """Initialize pattern matcher with compiled regex patterns."""
        # Character trait patterns
        self.trait_patterns = [
            re.compile(r'\b(?:is|was|seemed|appeared)\s+(?:very\s+)?(\w+(?:\s+and\s+\w+)?)\b', re.I),
            re.compile(r'\b(?:a|an)\s+(\w+)\s+(?:person|character|individual)\b', re.I),
            re.compile(r'\b(?:personality|demeanor|manner|attitude)\s+(?:was|is)\s+(\w+)\b', re.I),
        ]
        
        # Motivation patterns
        self.motivation_patterns = [
            re.compile(r'\b(?:wanted|desired|sought|craved|yearned)\s+(?:to\s+)?(.+?)(?:\.|,|;)', re.I),
            re.compile(r'\b(?:goal|objective|purpose|mission)\s+(?:was|is)\s+(?:to\s+)?(.+?)(?:\.|,|;)', re.I),
            re.compile(r'\b(?:driven|motivated)\s+by\s+(.+?)(?:\.|,|;)', re.I),
        ]
        
        # Fear patterns
        self.fear_patterns = [
            re.compile(r'\b(?:feared|afraid of|terrified of|scared of)\s+(.+?)(?:\.|,|;)', re.I),
            re.compile(r'\b(?:phobia|fear)\s+of\s+(.+?)(?:\.|,|;)', re.I),
        ]
        
        # Background patterns
        self.background_patterns = [
            re.compile(r'\b(?:was|had been)\s+(?:a|an)\s+(\w+(?:\s+\w+)?)\s+(?:before|previously|once)', re.I),
            re.compile(r'\b(?:former|ex-|retired)\s+(\w+(?:\s+\w+)?)\b', re.I),
            re.compile(r'\b(?:background|history)\s+(?:as|in)\s+(\w+(?:\s+\w+)?)\b', re.I),
        ]
        
        # NPC occupation patterns
        self.occupation_patterns = [
            re.compile(r'\b(?:worked as|employed as|served as)\s+(?:a|an)\s+(\w+(?:\s+\w+)?)\b', re.I),
            re.compile(r'\bthe\s+(\w+(?:\s+\w+)?)\s+(?:said|replied|answered|asked)\b', re.I),
            re.compile(r'\b(?:profession|occupation|job|trade)\s+(?:was|is)\s+(\w+(?:\s+\w+)?)\b', re.I),
        ]
        
        # Location patterns
        self.location_patterns = [
            re.compile(r'\b(?:in|at|near|by)\s+the\s+(\w+(?:\s+\w+){0,2})\b', re.I),
            re.compile(r'\b(?:traveled|went|arrived)\s+(?:to|at)\s+(\w+(?:\s+\w+){0,2})\b', re.I),
            re.compile(r'\b(?:city|town|village|port|station|outpost)\s+(?:of|called|named)\s+(\w+(?:\s+\w+)?)\b', re.I),
        ]
        
        # Item/weapon patterns
        self.item_patterns = [
            re.compile(r'\b(?:wielded|carried|held|bore)\s+(?:a|an|the)\s+(\w+(?:\s+\w+)?)\b', re.I),
            re.compile(r'\b(\w+(?:\s+\w+)?)\s+(?:sword|blade|gun|rifle|pistol|weapon)\b', re.I),
            re.compile(r'\b(?:armor|shield|helm|gauntlet)\s+(?:of|made from)\s+(\w+)\b', re.I),
        ]
        
        # Name patterns
        self.name_patterns = [
            re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?:said|replied|answered)', re.I),
            re.compile(r'\bcalled\s+(?:him|her|them)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'),
            re.compile(r'\bname\s+(?:was|is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'),
        ]
    
    def extract_from_text(self, text: str) -> ExtractedContent:
        """Extract TTRPG content from text using pattern matching."""
        content = ExtractedContent()
        
        # Split text into sentences for better processing
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue
            
            # Extract character traits
            for pattern in self.trait_patterns:
                matches = pattern.findall(sentence)
                for match in matches:
                    if isinstance(match, tuple):
                        match = ' '.join(match)
                    trait = self._clean_extracted_text(match)
                    if trait and self._is_valid_trait(trait):
                        content.character_traits.add(trait)
            
            # Extract motivations
            for pattern in self.motivation_patterns:
                matches = pattern.findall(sentence)
                for match in matches:
                    motivation = self._clean_extracted_text(match)
                    if motivation and len(motivation) > 5:
                        content.motivations.add(motivation[:100])  # Limit length
            
            # Extract fears
            for pattern in self.fear_patterns:
                matches = pattern.findall(sentence)
                for match in matches:
                    fear = self._clean_extracted_text(match)
                    if fear and len(fear) > 3:
                        content.fears.add(fear[:50])
            
            # Extract backgrounds
            for pattern in self.background_patterns:
                matches = pattern.findall(sentence)
                for match in matches:
                    background = self._clean_extracted_text(match)
                    if background and self._is_valid_occupation(background):
                        content.backgrounds.add(background)
            
            # Extract NPC occupations
            for pattern in self.occupation_patterns:
                matches = pattern.findall(sentence)
                for match in matches:
                    occupation = self._clean_extracted_text(match)
                    if occupation and self._is_valid_occupation(occupation):
                        content.npc_occupations.add(occupation)
            
            # Extract locations
            for pattern in self.location_patterns:
                matches = pattern.findall(sentence)
                for match in matches:
                    location = self._clean_extracted_text(match)
                    if location and self._is_valid_location(location):
                        content.locations.add(location)
            
            # Extract items and weapons
            for pattern in self.item_patterns:
                matches = pattern.findall(sentence)
                for match in matches:
                    item = self._clean_extracted_text(match)
                    if item and self._is_valid_item(item):
                        if any(weapon_word in item.lower() for weapon_word in 
                               ['sword', 'blade', 'gun', 'rifle', 'pistol', 'dagger', 'axe', 'bow']):
                            content.weapons.add(item)
                        elif any(armor_word in item.lower() for armor_word in 
                                ['armor', 'shield', 'helm', 'mail', 'plate', 'leather']):
                            content.armor.add(item)
                        else:
                            content.items.add(item)
            
            # Extract character names
            for pattern in self.name_patterns:
                matches = pattern.findall(sentence)
                for match in matches:
                    name = self._clean_extracted_text(match)
                    if name and self._is_valid_name(name):
                        content.character_names.add(name)
        
        return content
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common articles and prepositions at the start
        text = re.sub(r'^(?:the|a|an|to|in|at|on|of|for)\s+', '', text, flags=re.I)
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        return text.strip()
    
    def _is_valid_trait(self, trait: str) -> bool:
        """Check if extracted trait is valid."""
        return (
            len(trait) > 2 and 
            trait.lower() not in self._INVALID_TRAITS and
            not trait.isdigit()
        )
    
    def _is_valid_occupation(self, occupation: str) -> bool:
        """Check if extracted occupation is valid."""
        return (
            len(occupation) > 3 and
            occupation.lower() not in self._INVALID_OCCUPATIONS and
            not occupation.isdigit() and
            len(occupation.split()) <= 3  # Limit to 3 words
        )
    
    def _is_valid_location(self, location: str) -> bool:
        """Check if extracted location is valid."""
        return (
            len(location) > 2 and
            location.lower() not in self._INVALID_LOCATIONS and
            not location.isdigit() and
            len(location.split()) <= 4  # Limit to 4 words
        )
    
    def _is_valid_item(self, item: str) -> bool:
        """Check if extracted item is valid."""
        return (
            len(item) > 2 and
            not item.isdigit() and
            len(item.split()) <= 4  # Limit to 4 words
        )
    
    def _is_valid_name(self, name: str) -> bool:
        """Check if extracted name is valid."""
        # Basic validation for names
        return (
            len(name) > 2 and
            len(name) < 50 and
            not name.isdigit() and
            name[0].isupper() and
            len(name.split()) <= 3  # Limit to 3 words (first, middle, last)
        )


class EbookExtractor:
    """Main class for extracting TTRPG content from ebooks."""
    
    def __init__(self):
        """Initialize the ebook extractor."""
        self.pattern_matcher = ContentPatternMatcher()
        self.extracted_content = ExtractedContent()
    
    def extract_from_epub(self, epub_path: Path) -> ExtractedContent:
        """Extract content from an EPUB file."""
        content = ExtractedContent()
        
        try:
            book = epub.read_epub(str(epub_path))
            
            # Extract text from all items in the EPUB
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_body_content(), 'html.parser')
                    text = soup.get_text()
                    
                    # Extract content using pattern matching
                    extracted = self.pattern_matcher.extract_from_text(text)
                    content.merge(extracted)
                    
                    # Also extract from specific HTML elements
                    self._extract_from_html_structure(soup, content)
            
            logger.info(f"Extracted content from EPUB: {epub_path.name}")
            
        except Exception as e:
            logger.error(f"Error processing EPUB {epub_path.name}: {e}")
        
        return content
    
    def extract_from_mobi(self, mobi_path: Path) -> ExtractedContent:
        """Extract content from a MOBI file."""
        content = ExtractedContent()
        
        try:
            # Extract MOBI content
            tempdir = Path("/tmp") / f"mobi_extract_{mobi_path.stem}"
            tempdir.mkdir(exist_ok=True)
            
            # Use mobi library to extract
            book = mobi.Mobi(str(mobi_path))
            book.extract(str(tempdir))
            
            # Find and process extracted HTML files
            for html_file in tempdir.glob("*.html"):
                with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                    text = soup.get_text()
                    
                    # Extract content using pattern matching
                    extracted = self.pattern_matcher.extract_from_text(text)
                    content.merge(extracted)
                    
                    # Also extract from specific HTML elements
                    self._extract_from_html_structure(soup, content)
            
            # Clean up temp directory
            import shutil
            shutil.rmtree(tempdir, ignore_errors=True)
            
            logger.info(f"Extracted content from MOBI: {mobi_path.name}")
            
        except Exception as e:
            logger.error(f"Error processing MOBI {mobi_path.name}: {e}")
        
        return content
    
    def _extract_from_html_structure(self, soup: BeautifulSoup, content: ExtractedContent) -> None:
        """Extract content from specific HTML structures."""
        # Look for character descriptions in specific tags
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text = tag.get_text().strip()
            if text and len(text) > 2:
                # Headers often contain character names or location names
                if text[0].isupper() and len(text.split()) <= 3:
                    if any(word in text.lower() for word in ['chapter', 'part', 'section', 'book']):
                        continue
                    content.character_names.add(text)
        
        # Look for lists that might contain traits, items, etc.
        for ul in soup.find_all('ul'):
            for li in ul.find_all('li'):
                text = li.get_text().strip()
                if text and len(text) > 3:
                    # Simple heuristic: short list items might be traits or items
                    if len(text) < 50:
                        words = text.split()
                        if len(words) <= 3:
                            content.character_traits.add(text)
                        elif any(item_word in text.lower() for item_word in 
                                ['sword', 'armor', 'potion', 'scroll', 'ring', 'amulet']):
                            content.items.add(text)
        
        # Look for definition lists
        for dl in soup.find_all('dl'):
            for dt in dl.find_all('dt'):
                term = dt.get_text().strip()
                if term and len(term) > 2:
                    # Definition terms might be character names, locations, or items
                    if term[0].isupper():
                        content.character_names.add(term)
    
    def process_directory(self, directory: Path, file_pattern: str = "*.epub") -> ExtractedContent:
        """Process all ebooks in a directory."""
        total_content = ExtractedContent()
        
        for ebook_path in directory.glob(file_pattern):
            if ebook_path.suffix.lower() == '.epub':
                content = self.extract_from_epub(ebook_path)
            elif ebook_path.suffix.lower() == '.mobi':
                content = self.extract_from_mobi(ebook_path)
            else:
                continue
            
            total_content.merge(content)
        
        # Clean and filter the total content
        total_content.filter_and_clean()
        
        return total_content
    
    def save_extracted_content(self, content: ExtractedContent, output_path: Path) -> None:
        """Save extracted content to a JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(content.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved extracted content to {output_path}")
    
    def load_extracted_content(self, input_path: Path) -> ExtractedContent:
        """Load extracted content from a JSON file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        content = ExtractedContent()
        for field_name, values in data.items():
            if hasattr(content, field_name):
                setattr(content, field_name, set(values))
        
        return content


def main():
    """Main function to run the ebook extraction."""
    # Initialize extractor
    extractor = EbookExtractor()
    
    # Process EPUB files
    epub_dir = Path("./sample_epubs")
    if epub_dir.exists():
        logger.info("Processing EPUB files...")
        epub_content = extractor.process_directory(epub_dir, "*.epub")
        extractor.extracted_content.merge(epub_content)
    
    # Process MOBI files
    mobi_dir = Path("./sample_mobis")
    if mobi_dir.exists():
        logger.info("Processing MOBI files...")
        mobi_content = extractor.process_directory(mobi_dir, "*.mobi")
        extractor.extracted_content.merge(mobi_content)
    
    # Clean and filter final content
    extractor.extracted_content.filter_and_clean()
    
    # Save extracted content
    output_path = Path("extracted_content/ttrpg_content.json")
    extractor.save_extracted_content(extractor.extracted_content, output_path)
    
    # Print summary
    logger.info("\n=== Extraction Summary ===")
    for field_name in extractor.extracted_content.__dataclass_fields__:
        items = getattr(extractor.extracted_content, field_name)
        if items:
            logger.info(f"{field_name}: {len(items)} items")
            # Show a few examples
            examples = list(items)[:3]
            for example in examples:
                logger.info(f"  - {example[:50]}...")
    
    return extractor.extracted_content


if __name__ == "__main__":
    content = main()