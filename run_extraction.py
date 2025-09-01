#!/usr/bin/env python3
"""Main script to run the complete ebook extraction and content enrichment pipeline.

This script:
1. Extracts TTRPG content from EPUB and MOBI files
2. Enriches the extracted content with categories and metadata
3. Updates the character generation system with the new content
4. Demonstrates the enhanced character generation capabilities
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from content_enrichment import ContentEnricher
from ebook_extraction import EbookExtractor
from character_generation.enhanced_generator import (
    EnhancedCharacterGenerator,
    EnrichedContentManager,
)
from character_generation.models import CharacterClass, NPCRole, TTRPGGenre

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('extraction.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


def run_extraction(epub_dir: Path, mobi_dir: Path, output_dir: Path) -> Path:
    """Run the ebook extraction process.
    
    Args:
        epub_dir: Directory containing EPUB files
        mobi_dir: Directory containing MOBI files
        output_dir: Directory to save extracted content
        
    Returns:
        Path to the extracted content file
    """
    logger.info("=== Starting Ebook Extraction ===")
    
    # Initialize extractor
    extractor = EbookExtractor()
    
    # Process EPUB files
    if epub_dir.exists():
        logger.info(f"Processing EPUB files from {epub_dir}")
        epub_content = extractor.process_directory(epub_dir, "*.epub")
        extractor.extracted_content.merge(epub_content)
        logger.info(f"Processed {len(list(epub_dir.glob('*.epub')))} EPUB files")
    else:
        logger.warning(f"EPUB directory not found: {epub_dir}")
    
    # Process MOBI files
    if mobi_dir.exists():
        logger.info(f"Processing MOBI files from {mobi_dir}")
        mobi_content = extractor.process_directory(mobi_dir, "*.mobi")
        extractor.extracted_content.merge(mobi_content)
        logger.info(f"Processed {len(list(mobi_dir.glob('*.mobi')))} MOBI files")
    else:
        logger.warning(f"MOBI directory not found: {mobi_dir}")
    
    # Clean and filter final content
    extractor.extracted_content.filter_and_clean()
    
    # Save extracted content
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ttrpg_content.json"
    extractor.save_extracted_content(extractor.extracted_content, output_path)
    
    # Print extraction summary
    logger.info("\n=== Extraction Summary ===")
    for field_name in extractor.extracted_content.__dataclass_fields__:
        items = getattr(extractor.extracted_content, field_name)
        if items:
            logger.info(f"{field_name}: {len(items)} items extracted")
    
    return output_path


def run_enrichment(extracted_path: Path, output_dir: Path) -> Path:
    """Run the content enrichment process.
    
    Args:
        extracted_path: Path to extracted content JSON
        output_dir: Directory to save enriched content
        
    Returns:
        Path to the enriched content file
    """
    logger.info("\n=== Starting Content Enrichment ===")
    
    # Load extracted content
    from ebook_extraction import EbookExtractor
    extractor = EbookExtractor()
    extracted_content = extractor.load_extracted_content(extracted_path)
    
    # Enrich the content
    enricher = ContentEnricher()
    enriched_content = enricher.enrich_extracted_content(extracted_content)
    
    # Save enriched content
    output_path = output_dir / "enriched_content.json"
    enriched_content.save(output_path)
    
    # Print enrichment summary
    logger.info("\n=== Enrichment Summary ===")
    logger.info(f"Enriched Traits: {len(enriched_content.traits)}")
    logger.info(f"Enriched Backgrounds: {len(enriched_content.backgrounds)}")
    logger.info(f"Enriched Motivations: {len(enriched_content.motivations)}")
    logger.info(f"Story Hooks: {len(enriched_content.story_hooks)}")
    logger.info(f"World Elements: {len(enriched_content.world_elements)}")
    logger.info(f"Character Names: {len(enriched_content.character_names)}")
    logger.info(f"Weapons: {len(enriched_content.weapons)}")
    logger.info(f"Items: {len(enriched_content.items)}")
    
    return output_path


def demonstrate_enhanced_generation(enriched_path: Path) -> None:
    """Demonstrate the enhanced character generation capabilities.
    
    Args:
        enriched_path: Path to enriched content JSON
    """
    logger.info("\n=== Demonstrating Enhanced Character Generation ===")
    
    # Initialize the enhanced generator with the enriched content
    content_manager = EnrichedContentManager(enriched_path)
    generator = EnhancedCharacterGenerator(content_manager)
    
    # Generate characters for different genres
    genres = [
        TTRPGGenre.FANTASY,
        TTRPGGenre.SCI_FI,
        TTRPGGenre.CYBERPUNK,
        TTRPGGenre.POST_APOCALYPTIC
    ]
    
    for genre in genres:
        logger.info(f"\n--- {genre.value.upper()} Character ---")
        
        # Generate a character
        character = generator.generate_character(
            genre=genre,
            use_enriched_content=True
        )
        
        logger.info(f"Name: {character.name}")
        logger.info(f"Class: {character.get_class_name()}")
        logger.info(f"Race: {character.get_race_name()}")
        logger.info(f"Background: {character.backstory.background}")
        logger.info(f"Motivation: {character.backstory.motivation}")
        
        if character.backstory.personality_traits:
            logger.info(f"Personality: {', '.join(character.backstory.personality_traits[:2])}")
        
        if character.backstory.goals:
            logger.info(f"Goals: {', '.join(character.backstory.goals[:2])}")
        
        if character.equipment.weapons:
            logger.info(f"Weapons: {', '.join(character.equipment.weapons[:2])}")
    
    # Generate an NPC party
    logger.info("\n--- NPC Generation ---")
    
    for i in range(3):
        npc = generator.generate_npc_with_enriched_content(
            genre=TTRPGGenre.FANTASY,
            importance="Supporting" if i == 0 else "Minor"
        )
        
        logger.info(f"\nNPC {i+1}: {npc.name}")
        logger.info(f"  Role: {npc.get_role_name()}")
        logger.info(f"  Location: {npc.location}")
        logger.info(f"  Motivation: {npc.backstory.motivation}")
        
        if npc.secrets:
            logger.info(f"  Secret: {npc.secrets[0]}")
    
    # Generate a party with story hook
    logger.info("\n--- Party Generation with Story Hook ---")
    
    party, story_hook = generator.create_party_with_story_hook(
        party_size=4,
        genre=TTRPGGenre.FANTASY
    )
    
    logger.info(f"Story Hook: {story_hook.get('title', 'Unknown Quest')}")
    logger.info(f"Type: {story_hook.get('hook_type', 'quest')}")
    logger.info(f"Difficulty: {story_hook.get('difficulty', 'moderate')}")
    
    logger.info("\nParty Members:")
    for i, char in enumerate(party, 1):
        logger.info(f"{i}. {char.name}")
        logger.info(f"   Class: {char.get_class_name()}")
        logger.info(f"   Race: {char.get_race_name()}")
        
        if char.backstory.relationships:
            rel = char.backstory.relationships[0]
            logger.info(f"   Connection: {rel['relationship']} with {rel['name']}")


def main():
    """Main function to run the complete pipeline."""
    parser = argparse.ArgumentParser(
        description="Extract and enrich TTRPG content from ebooks"
    )
    parser.add_argument(
        "--epub-dir",
        type=Path,
        default=Path("/home/svnbjrn/code/phase12/sample_epubs"),
        help="Directory containing EPUB files"
    )
    parser.add_argument(
        "--mobi-dir",
        type=Path,
        default=Path("/home/svnbjrn/code/phase12/sample_mobis"),
        help="Directory containing MOBI files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("extracted_content"),
        help="Directory to save extracted and enriched content"
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip extraction and use existing extracted content"
    )
    parser.add_argument(
        "--skip-enrichment",
        action="store_true",
        help="Skip enrichment and use existing enriched content"
    )
    parser.add_argument(
        "--skip-demo",
        action="store_true",
        help="Skip the demonstration of enhanced character generation"
    )
    
    args = parser.parse_args()
    
    try:
        # Run extraction
        if not args.skip_extraction:
            extracted_path = run_extraction(
                args.epub_dir,
                args.mobi_dir,
                args.output_dir
            )
        else:
            extracted_path = args.output_dir / "ttrpg_content.json"
            if not extracted_path.exists():
                logger.error(f"Extracted content not found at {extracted_path}")
                logger.error("Please run extraction first or remove --skip-extraction")
                return 1
        
        # Run enrichment
        if not args.skip_enrichment:
            enriched_path = run_enrichment(extracted_path, args.output_dir)
        else:
            enriched_path = args.output_dir / "enriched_content.json"
            if not enriched_path.exists():
                logger.error(f"Enriched content not found at {enriched_path}")
                logger.error("Please run enrichment first or remove --skip-enrichment")
                return 1
        
        # Demonstrate enhanced generation
        if not args.skip_demo:
            demonstrate_enhanced_generation(enriched_path)
        
        logger.info("\n=== Pipeline Complete ===")
        logger.info(f"Extracted content saved to: {args.output_dir / 'ttrpg_content.json'}")
        logger.info(f"Enriched content saved to: {args.output_dir / 'enriched_content.json'}")
        logger.info("\nThe character generation system has been enhanced with the extracted content.")
        logger.info("You can now use the EnhancedCharacterGenerator class to create rich, diverse characters.")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())