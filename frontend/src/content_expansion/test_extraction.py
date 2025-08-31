#!/usr/bin/env python3
"""
Test script for the PDF content extraction system.

This script provides basic testing and verification of the extraction pipeline.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.content_expansion.pdf_analyzer import PDFAnalyzer
from src.content_expansion.genre_classifier import GenreClassifier
from src.content_expansion.models import TTRPGGenre


def test_genre_classifier():
    """Test the genre classification functionality."""
    print("Testing Genre Classifier...")
    print("-" * 40)
    
    classifier = GenreClassifier()
    
    test_cases = [
        ("Dungeons & Dragons Player's Handbook", None, TTRPGGenre.FANTASY),
        ("Cyberpunk 2020", None, TTRPGGenre.CYBERPUNK),
        ("Call of Cthulhu", None, TTRPGGenre.COSMIC_HORROR),
        ("Fallout RPG", None, TTRPGGenre.POST_APOCALYPTIC),
        ("Stars Without Number", None, TTRPGGenre.SPACE_OPERA),
    ]
    
    # Test with sample text
    fantasy_text = """
    The wizard casts a spell using their arcane magic. The dragon breathes fire
    upon the knight's shield. Elves and dwarves gather in the tavern to discuss
    their quest to defeat the necromancer and retrieve the ancient artifact.
    """
    
    cyberpunk_text = """
    The netrunner jacks into cyberspace through their neural implant. The megacorp's
    ICE defenses are strong, but the hacker's custom daemon might breach the firewall.
    Chrome augmentations gleam under the neon lights of the sprawl.
    """
    
    # Test title classification
    for title, text, expected in test_cases:
        genre = classifier.classify_by_title(title)
        status = "✓" if genre == expected else "✗"
        print(f"{status} Title: '{title}' -> {genre.name if genre else 'None'}")
    
    # Test content classification
    print("\nContent Classification:")
    genre, confidence = classifier.classify_by_content(fantasy_text)
    print(f"Fantasy text -> {genre.name} (confidence: {confidence:.2f})")
    
    genre, confidence = classifier.classify_by_content(cyberpunk_text)
    print(f"Cyberpunk text -> {genre.name} (confidence: {confidence:.2f})")
    
    print("\n✓ Genre classifier tests complete\n")


def test_single_pdf(pdf_path: str):
    """Test extraction from a single PDF."""
    print(f"Testing PDF Extraction: {pdf_path}")
    print("-" * 40)
    
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        print(f"✗ PDF file not found: {pdf_path}")
        return
    
    analyzer = PDFAnalyzer(
        output_dir="./test_output",
        cache_dir="./test_cache",
        use_multiprocessing=False
    )
    
    print(f"Analyzing {pdf_file.name}...")
    result = analyzer.analyze_single_pdf(pdf_file)
    
    if result:
        print(f"✓ Successfully extracted content from {pdf_file.name}")
        print("\nExtraction Summary:")
        summary = result.get_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        print(f"\nGenre: {result.genre.name}")
        print(f"Extraction Method: {result.extraction_metadata.get('extraction_method', 'unknown')}")
        
        # Show sample content
        if result.races:
            print(f"\nSample Race: {result.races[0].name}")
        if result.classes:
            print(f"Sample Class: {result.classes[0].name}")
        if result.equipment:
            print(f"Sample Equipment: {result.equipment[0].name}")
    else:
        print(f"✗ Failed to extract content from {pdf_file.name}")
    
    print()


def test_batch_processing(pdf_dir: str, max_files: int = 5):
    """Test batch processing of PDFs."""
    print(f"Testing Batch Processing: {pdf_dir}")
    print("-" * 40)
    
    pdf_directory = Path(pdf_dir)
    
    if not pdf_directory.exists():
        print(f"✗ Directory not found: {pdf_dir}")
        return
    
    # Get first few PDFs for testing
    pdf_files = list(pdf_directory.glob("*.pdf"))[:max_files]
    
    if not pdf_files:
        print(f"✗ No PDF files found in {pdf_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDFs to test")
    
    analyzer = PDFAnalyzer(
        output_dir="./test_output",
        cache_dir="./test_cache",
        use_multiprocessing=False  # Use sequential for testing
    )
    
    successful = 0
    failed = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing {pdf_file.name}...")
        
        try:
            result = analyzer.analyze_single_pdf(pdf_file)
            if result:
                successful += 1
                print(f"  ✓ Genre: {result.genre.name}")
                summary = result.get_summary()
                print(f"  ✓ Extracted: {sum(summary.values())} items")
            else:
                failed += 1
                print(f"  ✗ Extraction failed")
        except Exception as e:
            failed += 1
            print(f"  ✗ Error: {e}")
    
    print(f"\nBatch Processing Results:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Success Rate: {successful/len(pdf_files)*100:.1f}%")


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test PDF content extraction")
    parser.add_argument("--classifier", action="store_true",
                       help="Test genre classifier")
    parser.add_argument("--single", help="Test single PDF extraction")
    parser.add_argument("--batch", help="Test batch processing on directory")
    parser.add_argument("--max-files", type=int, default=5,
                       help="Maximum files for batch test")
    parser.add_argument("--all", action="store_true",
                       help="Run all tests")
    
    args = parser.parse_args()
    
    print("="*60)
    print("PDF CONTENT EXTRACTION SYSTEM TEST")
    print("="*60)
    print(f"Test started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    if args.all or args.classifier:
        test_genre_classifier()
    
    if args.single:
        test_single_pdf(args.single)
    elif args.all:
        # Test with first PDF in sample directory if it exists
        sample_dir = Path("/home/svnbjrn/code/phase12/sample_pdfs")
        if sample_dir.exists():
            pdfs = list(sample_dir.glob("*.pdf"))
            if pdfs:
                test_single_pdf(str(pdfs[0]))
    
    if args.batch:
        test_batch_processing(args.batch, args.max_files)
    elif args.all:
        sample_dir = "/home/svnbjrn/code/phase12/sample_pdfs"
        if Path(sample_dir).exists():
            test_batch_processing(sample_dir, 3)
    
    if not any([args.classifier, args.single, args.batch, args.all]):
        print("No tests specified. Use --help for options.")
        print("\nQuick test command examples:")
        print("  python test_extraction.py --classifier")
        print("  python test_extraction.py --single /path/to/file.pdf")
        print("  python test_extraction.py --batch /home/svnbjrn/code/phase12/sample_pdfs")
        print("  python test_extraction.py --all")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()