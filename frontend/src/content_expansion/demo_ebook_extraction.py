#!/usr/bin/env python3
"""
Demonstration script for ebook extraction capabilities.

This script shows how to use the ebook extraction modules to process
EPUB, MOBI, and other ebook formats.
"""

import sys
from pathlib import Path
from unified_analyzer import UnifiedEbookAnalyzer, EbookFormat


def demo_extraction():
    """Demonstrate ebook extraction capabilities."""
    print("="*60)
    print("Ebook Extraction Module Demo")
    print("="*60)
    
    # Initialize the analyzer
    analyzer = UnifiedEbookAnalyzer(cache_dir=Path("./extraction_cache"))
    
    # Show supported formats
    print("\nSupported ebook formats:")
    for fmt in analyzer.get_supported_formats():
        print(f"  - .{fmt}")
    
    print("\n" + "="*60)
    print("Usage Examples:")
    print("="*60)
    
    print("""
1. Process a complete ebook:
   
   result = analyzer.process_ebook("path/to/book.epub")
   print(f"Title: {result['metadata']['title']}")
   print(f"Word count: {result['analysis']['statistics']['word_count']}")
   print(f"Reading time: {result['analysis']['estimated_reading_time']} minutes")
   
2. Quick text extraction:
   
   text = analyzer.quick_extract_text("path/to/book.mobi")
   print(f"Extracted {len(text)} characters")
   
3. Extract metadata only:
   
   metadata = analyzer.extract_metadata("path/to/book.azw3")
   print(f"Author: {metadata['author']}")
   print(f"Title: {metadata['title']}")
   
4. Direct format-specific extraction:
   
   from epub_extractor import EPUBExtractor
   with EPUBExtractor("path/to/book.epub") as extractor:
       data = extractor.extract()
       for chapter in data['chapters']:
           print(f"Chapter: {chapter['title']}")
           
5. Content analysis:
   
   from ebook_analyzer import EbookAnalyzer
   analyzer = EbookAnalyzer()
   analysis = analyzer.analyze_content(text_content)
   print(f"Reading level: {analysis.reading_level}")
   print(f"Key themes: {', '.join(analysis.key_themes)}")
   summary = analyzer.generate_summary(text_content)
   print(f"Summary: {summary}")
    """)
    
    print("\n" + "="*60)
    print("To test with actual files:")
    print("="*60)
    print("""
1. Create a test_ebooks directory:
   mkdir test_ebooks
   
2. Add some ebook files (.epub, .mobi, .azw, .pdf)

3. Run the test script:
   python test_ebook_extraction.py
   
4. Or use this demo script with a file:
   python demo_ebook_extraction.py path/to/ebook.epub
    """)


def process_file(file_path: str):
    """Process a specific ebook file."""
    path = Path(file_path)
    
    if not path.exists():
        print(f"Error: File not found: {file_path}")
        return
    
    print(f"\nProcessing: {path.name}")
    print("="*60)
    
    try:
        analyzer = UnifiedEbookAnalyzer(cache_dir=Path("./extraction_cache"))
        
        # Detect format
        fmt = EbookFormat.from_extension(path)
        print(f"Format detected: {fmt.value}")
        
        # Process the ebook
        print("\nExtracting and analyzing content...")
        result = analyzer.process_ebook(path)
        
        # Display results
        print("\nMetadata:")
        print("-"*40)
        metadata = result['metadata']
        for key, value in metadata.items():
            if value and key not in ['contributors', 'subjects']:
                print(f"  {key.title()}: {value}")
        
        print("\nAnalysis:")
        print("-"*40)
        analysis = result['analysis']
        stats = analysis['statistics']
        print(f"  Word Count: {stats['word_count']:,}")
        print(f"  Sentence Count: {stats['sentence_count']:,}")
        print(f"  Paragraph Count: {stats['paragraph_count']:,}")
        print(f"  Unique Words: {stats['unique_words']:,}")
        print(f"  Lexical Diversity: {stats['lexical_diversity']:.2%}")
        print(f"  Reading Level: {analysis['reading_level']}")
        print(f"  Estimated Reading Time: {analysis['estimated_reading_time']} minutes")
        
        if analysis['key_themes']:
            print(f"\nKey Themes:")
            for theme in analysis['key_themes'][:5]:
                print(f"  - {theme}")
        
        if analysis['frequent_words']:
            print(f"\nTop Frequent Words:")
            for word, count in analysis['frequent_words'][:10]:
                print(f"  - {word}: {count}")
        
        if analysis['summary']:
            print(f"\nSummary:")
            print(f"  {analysis['summary'][:500]}...")
            
        print("\n✓ Processing complete!")
        
    except Exception as e:
        print(f"✗ Error processing file: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        # Process provided file
        file_path = sys.argv[1]
        process_file(file_path)
    else:
        # Show demo information
        demo_extraction()


if __name__ == "__main__":
    main()