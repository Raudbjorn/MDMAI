#!/usr/bin/env python
"""Simple test to verify pypdf 6.0.0 compatibility"""

import sys
import os
import tempfile

def test_pypdf_import():
    """Test basic import and version"""
    try:
        import pypdf
        print(f"✓ pypdf version: {pypdf.__version__}")
        assert pypdf.__version__.startswith("6.")
        return True
    except Exception as e:
        print(f"✗ Failed to import pypdf: {e}")
        return False

def test_pdf_creation():
    """Test creating a simple PDF"""
    try:
        from pypdf import PdfWriter, PdfReader
        from pypdf.generic import RectangleObject
        
        # Create a simple PDF
        writer = PdfWriter()
        page = writer.add_blank_page(width=200, height=200)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            writer.write(tmp)
            tmp_path = tmp.name
        
        # Read it back
        reader = PdfReader(tmp_path)
        assert len(reader.pages) == 1
        print("✓ PDF creation and reading works")
        
        # Clean up
        os.unlink(tmp_path)
        return True
    except Exception as e:
        print(f"✗ PDF operations failed: {e}")
        return False

def test_api_compatibility():
    """Test compatibility with existing API patterns"""
    try:
        from pypdf import PdfReader, PdfWriter
        
        # Check key classes exist
        assert hasattr(PdfReader, 'pages')
        assert hasattr(PdfWriter, 'add_page')
        assert hasattr(PdfWriter, 'write')
        print("✓ Key APIs are available")
        
        # Test that common patterns work
        writer = PdfWriter()
        writer.add_blank_page(100, 100)
        print("✓ Common patterns work")
        
        return True
    except Exception as e:
        print(f"✗ API compatibility check failed: {e}")
        return False

def test_metadata_handling():
    """Test PDF metadata operations"""
    try:
        from pypdf import PdfWriter
        
        writer = PdfWriter()
        writer.add_blank_page(100, 100)
        
        # Add metadata
        writer.add_metadata({
            '/Title': 'Test PDF',
            '/Author': 'Test Suite',
            '/Subject': 'Testing pypdf 6.0.0'
        })
        
        print("✓ Metadata handling works")
        return True
    except Exception as e:
        print(f"✗ Metadata handling failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing pypdf 6.0.0 Update")
    print("-" * 40)
    
    tests = [
        ("Import Test", test_pypdf_import),
        ("PDF Creation", test_pdf_creation),
        ("API Compatibility", test_api_compatibility),
        ("Metadata Handling", test_metadata_handling),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nRunning {name}...")
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 40)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("✓ All tests passed! pypdf 6.0.0 is compatible.")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Review the output above.")
        sys.exit(1)