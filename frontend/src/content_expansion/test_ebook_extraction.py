#!/usr/bin/env python3
"""
Test script for ebook extraction modules.

This script tests the functionality of the ebook extraction and analysis modules.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from epub_extractor import EPUBExtractor
from mobi_extractor import MOBIExtractor
from ebook_analyzer import EbookAnalyzer
from unified_analyzer import UnifiedEbookAnalyzer, EbookFormat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_epub_extraction(file_path: Optional[Path] = None):
    """Test EPUB extraction functionality."""
    print("\n" + "="*60)
    print("Testing EPUB Extraction")
    print("="*60)
    
    if not file_path:
        print("No EPUB file provided for testing")
        return
        
    try:
        with EPUBExtractor(file_path) as extractor:
            result = extractor.extract()
            
        print(f"✓ Successfully extracted EPUB: {file_path.name}")
        print(f"  - Title: {result['metadata'].get('title', 'Unknown')}")
        print(f"  - Author: {result['metadata'].get('author', 'Unknown')}")
        print(f"  - Chapters: {result['total_chapters']}")
        
        if result['chapters']:
            first_chapter = result['chapters'][0]
            print(f"  - First chapter title: {first_chapter['title']}")
            print(f"  - First chapter preview: {first_chapter['content'][:200]}...")
            
        return True
        
    except Exception as e:
        print(f"✗ Failed to extract EPUB: {e}")
        logger.exception("EPUB extraction failed")
        return False
        

def test_mobi_extraction(file_path: Optional[Path] = None):
    """Test MOBI extraction functionality."""
    print("\n" + "="*60)
    print("Testing MOBI Extraction")
    print("="*60)
    
    if not file_path:
        print("No MOBI file provided for testing")
        return
        
    try:
        with MOBIExtractor(file_path) as extractor:
            result = extractor.extract()
            
        print(f"✓ Successfully extracted MOBI: {file_path.name}")
        print(f"  - Title: {result['metadata'].get('title', 'Unknown')}")
        print(f"  - Author: {result['metadata'].get('author', 'Unknown')}")
        print(f"  - Encoding: {result['encoding']}")
        print(f"  - Content preview: {result['content'][:200]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to extract MOBI: {e}")
        logger.exception("MOBI extraction failed")
        return False
        

def test_content_analysis():
    """Test content analysis functionality."""
    print("\n" + "="*60)
    print("Testing Content Analysis")
    print("="*60)
    
    # Sample text for testing
    sample_text = """
    The quick brown fox jumps over the lazy dog. This is a sample text for testing
    the content analysis functionality. It contains multiple sentences and demonstrates
    various features of the analyzer.
    
    In this second paragraph, we continue with more text to ensure proper paragraph
    counting and statistical analysis. The analyzer should be able to extract meaningful
    insights from this content.
    
    Finally, we have a third paragraph with additional content. This helps test the
    robustness of our analysis algorithms and ensures they work correctly with
    multi-paragraph text.
    """
    
    try:
        analyzer = EbookAnalyzer()
        analysis = analyzer.analyze_content(sample_text)
        
        print("✓ Successfully analyzed content")
        print(f"  - Word count: {analysis.statistics.word_count}")
        print(f"  - Sentence count: {analysis.statistics.sentence_count}")
        print(f"  - Paragraph count: {analysis.statistics.paragraph_count}")
        print(f"  - Average word length: {analysis.statistics.average_word_length:.2f}")
        print(f"  - Average sentence length: {analysis.statistics.average_sentence_length:.2f}")
        print(f"  - Lexical diversity: {analysis.statistics.lexical_diversity:.2f}")
        print(f"  - Reading level: {analysis.reading_level}")
        print(f"  - Estimated reading time: {analysis.estimated_reading_time} minutes")
        
        if analysis.frequent_words:
            print(f"  - Top 5 frequent words: {analysis.frequent_words[:5]}")
            
        # Test summary generation
        summary = analyzer.generate_summary(sample_text, max_sentences=2)
        print(f"  - Summary: {summary}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to analyze content: {e}")
        logger.exception("Content analysis failed")
        return False
        

def test_unified_analyzer(file_path: Optional[Path] = None):
    """Test unified analyzer functionality."""
    print("\n" + "="*60)
    print("Testing Unified Analyzer")
    print("="*60)
    
    if not file_path:
        print("No file provided for testing")
        return
        
    try:
        analyzer = UnifiedEbookAnalyzer()
        
        # Test format detection
        detected_format = EbookFormat.from_extension(file_path)
        print(f"✓ Detected format: {detected_format.value}")
        
        # Test full processing
        result = analyzer.process_ebook(file_path)
        
        print(f"✓ Successfully processed ebook: {file_path.name}")
        print(f"  - Format: {result['format']}")
        print(f"  - Title: {result['metadata'].get('title', 'Unknown')}")
        print(f"  - Word count: {result['analysis']['statistics']['word_count']}")
        print(f"  - Reading level: {result['analysis']['reading_level']}")
        print(f"  - Reading time: {result['analysis']['estimated_reading_time']} minutes")
        
        if result['analysis']['key_themes']:
            print(f"  - Key themes: {', '.join(result['analysis']['key_themes'][:5])}")
            
        if result['analysis']['summary']:
            print(f"  - Summary preview: {result['analysis']['summary'][:200]}...")
            
        return True
        
    except Exception as e:
        print(f"✗ Failed unified analysis: {e}")
        logger.exception("Unified analysis failed")
        return False
        

def test_quick_extraction(file_path: Optional[Path] = None):
    """Test quick text extraction."""
    print("\n" + "="*60)
    print("Testing Quick Text Extraction")
    print("="*60)
    
    if not file_path:
        print("No file provided for testing")
        return
        
    try:
        analyzer = UnifiedEbookAnalyzer()
        text = analyzer.quick_extract_text(file_path)
        
        print(f"✓ Successfully extracted text from: {file_path.name}")
        print(f"  - Text length: {len(text)} characters")
        print(f"  - Preview: {text[:300]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed quick extraction: {e}")
        logger.exception("Quick extraction failed")
        return False
        

def main():
    """Main test execution."""
    print("\n" + "="*60)
    print("Ebook Extraction Module Tests")
    print("="*60)
    
    # Test with sample files if provided
    test_dir = Path(__file__).parent / "test_ebooks"
    
    # Run basic tests
    results = []
    
    # Test content analysis (doesn't need files)
    results.append(("Content Analysis", test_content_analysis()))
    
    # Test with actual files if available
    if test_dir.exists():
        # Look for test files
        epub_files = list(test_dir.glob("*.epub"))
        mobi_files = list(test_dir.glob("*.mobi")) + list(test_dir.glob("*.azw*"))
        
        if epub_files:
            results.append(("EPUB Extraction", test_epub_extraction(epub_files[0])))
            results.append(("Unified Analyzer (EPUB)", test_unified_analyzer(epub_files[0])))
            results.append(("Quick Extraction (EPUB)", test_quick_extraction(epub_files[0])))
            
        if mobi_files:
            results.append(("MOBI Extraction", test_mobi_extraction(mobi_files[0])))
            results.append(("Unified Analyzer (MOBI)", test_unified_analyzer(mobi_files[0])))
            results.append(("Quick Extraction (MOBI)", test_quick_extraction(mobi_files[0])))
    else:
        print(f"\nNote: No test_ebooks directory found at {test_dir}")
        print("Create this directory and add sample ebook files for full testing")
        
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
        
    print(f"\nTotal: {passed}/{total} tests passed")
    
    # Return exit code
    return 0 if passed == total else 1
    

if __name__ == "__main__":
    sys.exit(main())