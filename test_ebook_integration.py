#!/usr/bin/env python3
"""Test script for ebook processing integration.

This script tests the integration of EPUB and MOBI processing
into the existing MCP document processing pipeline.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.logging_config import get_logger
from src.pdf_processing.document_parser import UnifiedDocumentParser
from src.pdf_processing.pipeline import DocumentProcessingPipeline
from src.mcp_tools.document_tools import (
    ProcessDocumentTool,
    ProcessDocumentInput,
    ExtractDocumentTextTool,
    ExtractDocumentTextInput,
)

logger = get_logger(__name__)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_result(result: Dict[str, Any], indent: int = 0):
    """Print a result dictionary in a readable format."""
    spacing = "  " * indent
    for key, value in result.items():
        if isinstance(value, dict):
            print(f"{spacing}{key}:")
            print_result(value, indent + 1)
        elif isinstance(value, list) and len(value) > 0:
            print(f"{spacing}{key}: [{len(value)} items]")
        else:
            print(f"{spacing}{key}: {value}")


async def test_unified_parser():
    """Test the unified document parser with different formats."""
    print_section("Testing Unified Document Parser")
    
    parser = UnifiedDocumentParser()
    
    # Test supported formats
    print("\nSupported formats:", parser.get_supported_formats())
    
    # Test format detection
    test_files = [
        "example.pdf",
        "book.epub", 
        "novel.mobi",
        "kindle.azw3",
        "invalid.txt"
    ]
    
    print("\nFormat support check:")
    for file in test_files:
        supported = parser.is_format_supported(file)
        print(f"  {file}: {'✓ Supported' if supported else '✗ Not supported'}")
    
    return True


async def test_extract_text_tool():
    """Test the text extraction tool."""
    print_section("Testing Text Extraction Tool")
    
    tool = ExtractDocumentTextTool()
    
    # Create a mock test file (you can replace with actual test files)
    test_file = Path("test_document.pdf")
    
    if not test_file.exists():
        print(f"Skipping extraction test - {test_file} not found")
        print("To test extraction, place a test document at: test_document.pdf")
        return False
    
    try:
        input_data = ExtractDocumentTextInput(
            document_path=str(test_file),
            extract_tables=True,
            extract_metadata=True
        )
        
        result = await tool.execute(input_data)
        
        print("\nExtraction Results:")
        print(f"  Success: {result.success}")
        print(f"  Document Type: {result.document_type}")
        print(f"  Total Pages: {result.total_pages}")
        print(f"  Total Characters: {result.total_characters}")
        
        if result.metadata:
            print(f"  Metadata Keys: {list(result.metadata.keys())}")
        
        if result.tables:
            print(f"  Tables Found: {len(result.tables)}")
        
        return result.success
        
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False


async def test_process_document_tool():
    """Test the full document processing tool."""
    print_section("Testing Document Processing Tool")
    
    tool = ProcessDocumentTool()
    
    # Create a mock test file
    test_file = Path("test_rulebook.pdf")
    
    if not test_file.exists():
        print(f"Skipping processing test - {test_file} not found")
        print("To test processing, place a test document at: test_rulebook.pdf")
        return False
    
    try:
        input_data = ProcessDocumentInput(
            document_path=str(test_file),
            rulebook_name="Test Rulebook",
            system="Test System",
            source_type="rulebook",
            enable_adaptive_learning=False,
            skip_size_check=True
        )
        
        result = await tool.execute(input_data)
        
        print("\nProcessing Results:")
        print_result(result.dict(exclude_none=True))
        
        return result.status == "success"
        
    except Exception as e:
        print(f"Processing failed: {e}")
        return False


async def test_pipeline_backward_compatibility():
    """Test backward compatibility with existing PDF processing."""
    print_section("Testing Backward Compatibility")
    
    # Test that PDFProcessingPipeline alias still works
    from src.pdf_processing.pipeline import PDFProcessingPipeline
    
    pipeline = PDFProcessingPipeline(prompt_for_ollama=False)
    
    print("✓ PDFProcessingPipeline alias works")
    
    # Test that process_pdf method still exists
    if hasattr(pipeline, 'process_pdf'):
        print("✓ process_pdf method exists")
    else:
        print("✗ process_pdf method missing")
        return False
    
    # Test file validation for different formats
    test_paths = {
        "test.pdf": True,
        "test.epub": True,
        "test.mobi": True,
        "test.txt": False,
    }
    
    print("\nValidation tests:")
    for path, should_pass in test_paths.items():
        try:
            # Create the file temporarily for testing
            test_file = Path(path)
            test_file.touch()
            
            try:
                result = pipeline._validate_document_inputs(
                    str(test_file),
                    "Test",
                    "System",
                    "rulebook"
                )
                if should_pass:
                    print(f"  ✓ {path}: Correctly validated")
                else:
                    print(f"  ✗ {path}: Should have failed validation")
            except ValueError as e:
                if not should_pass:
                    print(f"  ✓ {path}: Correctly rejected")
                else:
                    print(f"  ✗ {path}: Should have passed validation")
            finally:
                test_file.unlink()  # Clean up
                
        except Exception as e:
            print(f"  ✗ {path}: Test error - {e}")
    
    return True


async def test_ebook_parser_dependencies():
    """Test that ebook parser can handle missing dependencies gracefully."""
    print_section("Testing Ebook Parser Dependencies")
    
    from src.pdf_processing.ebook_parser import EbookParser, EBOOKLIB_AVAILABLE, MOBI_AVAILABLE
    
    parser = EbookParser()
    
    print(f"ebooklib available: {EBOOKLIB_AVAILABLE}")
    print(f"mobi library available: {MOBI_AVAILABLE}")
    
    if not EBOOKLIB_AVAILABLE:
        print("\nNote: Install ebooklib for full EPUB support:")
        print("  pip install ebooklib")
    
    if not MOBI_AVAILABLE:
        print("\nNote: Install mobi for full MOBI support:")
        print("  pip install mobi")
    
    # Test that parser initializes even without dependencies
    print("\n✓ EbookParser initializes successfully")
    
    return True


async def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print(" EBOOK INTEGRATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Unified Parser", test_unified_parser),
        ("Ebook Parser Dependencies", test_ebook_parser_dependencies),
        ("Backward Compatibility", test_pipeline_backward_compatibility),
        ("Text Extraction Tool", test_extract_text_tool),
        ("Document Processing Tool", test_process_document_tool),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = "PASS" if result else "FAIL"
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}", exc_info=True)
            results[test_name] = "ERROR"
    
    # Print summary
    print_section("Test Summary")
    
    for test_name, status in results.items():
        symbol = "✓" if status == "PASS" else "✗"
        print(f"  {symbol} {test_name}: {status}")
    
    # Overall result
    all_passed = all(status == "PASS" for status in results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print(" ALL TESTS PASSED ✓")
    else:
        print(" SOME TESTS FAILED ✗")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    # Run tests
    success = asyncio.run(run_all_tests())
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)