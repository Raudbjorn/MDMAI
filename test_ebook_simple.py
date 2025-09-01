#!/usr/bin/env python3
"""Simple test for ebook parser functionality without full pipeline dependencies."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_ebook_parser_init():
    """Test that ebook parser can be imported and initialized."""
    print("Testing EbookParser import and initialization...")
    
    try:
        from src.pdf_processing.ebook_parser import EbookParser, EBOOKLIB_AVAILABLE, MOBI_AVAILABLE
        
        parser = EbookParser()
        print(f"✓ EbookParser imported successfully")
        print(f"  - ebooklib available: {EBOOKLIB_AVAILABLE}")
        print(f"  - mobi library available: {MOBI_AVAILABLE}")
        
        # Test file hash calculation
        test_file = Path("test_file.tmp")
        test_file.write_text("test content")
        
        try:
            hash_value = parser._calculate_file_hash(test_file)
            print(f"✓ File hash calculation works: {hash_value[:16]}...")
        finally:
            test_file.unlink()
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to import/initialize EbookParser: {e}")
        return False


def test_unified_parser():
    """Test the unified document parser."""
    print("\nTesting UnifiedDocumentParser...")
    
    try:
        from src.pdf_processing.document_parser import UnifiedDocumentParser
        
        parser = UnifiedDocumentParser()
        print("✓ UnifiedDocumentParser imported successfully")
        
        # Test supported formats
        formats = parser.get_supported_formats()
        print(f"✓ Supported formats: {', '.join(formats)}")
        
        # Test format checking
        test_files = [
            ("document.pdf", True),
            ("book.epub", True),
            ("novel.mobi", True),
            ("kindle.azw3", True),
            ("text.txt", False),
        ]
        
        print("\nFormat support tests:")
        for filename, expected in test_files:
            result = parser.is_format_supported(filename)
            if result == expected:
                print(f"  ✓ {filename}: {result}")
            else:
                print(f"  ✗ {filename}: Expected {expected}, got {result}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test UnifiedDocumentParser: {e}")
        return False


def test_backward_compatibility():
    """Test that backward compatibility is maintained."""
    print("\nTesting backward compatibility...")
    
    try:
        # Import original PDF parser
        from src.pdf_processing.pdf_parser import PDFParser
        
        pdf_parser = PDFParser()
        print("✓ PDFParser still imports correctly")
        
        # Check for expected methods
        if hasattr(pdf_parser, 'extract_text_from_pdf'):
            print("✓ extract_text_from_pdf method exists")
        else:
            print("✗ extract_text_from_pdf method missing")
            return False
        
        if hasattr(pdf_parser, 'extract_tables_as_markdown'):
            print("✓ extract_tables_as_markdown method exists")
        else:
            print("✗ extract_tables_as_markdown method missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        return False


def test_mcp_tools():
    """Test MCP tool definitions."""
    print("\nTesting MCP tools...")
    
    try:
        from src.mcp_tools.document_tools import (
            ProcessDocumentTool,
            ExtractDocumentTextTool,
            GetSupportedFormatsTool,
            list_tools,
        )
        
        print("✓ MCP tools imported successfully")
        
        # List available tools
        tools = list_tools()
        print(f"✓ Found {len(tools)} MCP tools:")
        for tool in tools:
            print(f"  - {tool['name']}: {tool['description']}")
        
        # Test tool schema generation
        process_tool = ProcessDocumentTool()
        schema = process_tool.get_schema()
        
        if 'name' in schema and 'description' in schema:
            print("✓ Tool schema generation works")
        else:
            print("✗ Tool schema incomplete")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ MCP tools test failed: {e}")
        return False


def main():
    """Run all simple tests."""
    print("=" * 60)
    print(" EBOOK INTEGRATION SIMPLE TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("EbookParser Initialization", test_ebook_parser_init),
        ("Unified Document Parser", test_unified_parser),
        ("Backward Compatibility", test_backward_compatibility),
        ("MCP Tool Definitions", test_mcp_tools),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 60)
    print(" TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        symbol = "✓" if passed else "✗"
        status = "PASS" if passed else "FAIL"
        print(f"  {symbol} {test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print(" ALL TESTS PASSED ✓")
    else:
        print(" SOME TESTS FAILED ✗")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())