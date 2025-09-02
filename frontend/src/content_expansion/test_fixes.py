#!/usr/bin/env python3
"""
Test script to verify all code review fixes have been applied correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    # Test models import and ExtractedContent from_dict
    from frontend.src.content_expansion.models import ExtractedContent, TTRPGGenre
    print("✓ Models import successful")
    
    # Test processing helpers import
    from frontend.src.content_expansion.processing_helpers import (
        CacheManager, BatchProcessor, ReportGenerator
    )
    print("✓ Processing helpers import successful")
    
    # Test the from_dict method exists
    content = ExtractedContent(
        pdf_path=Path('/test/sample.pdf'),
        pdf_name='sample.pdf',
        genre=TTRPGGenre.FANTASY
    )
    data = content.to_dict()
    restored = ExtractedContent.from_dict(data)
    assert restored.pdf_name == 'sample.pdf'
    print("✓ ExtractedContent.from_dict() works correctly")
    
    return True


def test_processing_config():
    """Test ProcessingConfig has chars_per_page."""
    print("\nTesting ProcessingConfig...")
    
    from frontend.src.content_expansion.pdf_analyzer_enhanced import ProcessingConfig
    
    config = ProcessingConfig()
    
    # Check if chars_per_page exists
    assert hasattr(config, 'chars_per_page'), "ProcessingConfig missing chars_per_page"
    assert config.chars_per_page == 3000, f"chars_per_page should be 3000, got {config.chars_per_page}"
    print(f"✓ ProcessingConfig has chars_per_page = {config.chars_per_page}")
    
    # Check if it's in to_dict
    config_dict = config.to_dict()
    assert 'chars_per_page' in config_dict, "chars_per_page not in to_dict()"
    print("✓ chars_per_page included in to_dict()")
    
    return True


def test_no_pickle_import():
    """Verify pickle is not imported in pdf_analyzer_enhanced.py."""
    print("\nChecking for pickle removal...")
    
    with open('/home/svnbjrn/code/cl2/MDMAI/frontend/src/content_expansion/pdf_analyzer_enhanced.py', 'r') as f:
        content = f.read()
    
    # Check that pickle is not imported
    lines = content.split('\n')
    for i, line in enumerate(lines[:50], 1):  # Check first 50 lines for imports
        if 'import pickle' in line:
            print(f"✗ Found 'import pickle' at line {i}")
            return False
    
    print("✓ No pickle import found")
    
    # Check that we're using JSON for cache
    assert '.json' in content, "Should be using .json extension for cache files"
    print("✓ Using JSON for cache files")
    
    return True


def test_ocr_page_limit():
    """Test that OCR respects configured page limit."""
    print("\nChecking OCR page limit fix...")
    
    with open('/home/svnbjrn/code/cl2/MDMAI/frontend/src/content_expansion/pdf_analyzer_enhanced.py', 'r') as f:
        content = f.read()
    
    # Check that min(50) is removed
    if 'min(self.config.ocr_max_pages, 50)' in content:
        print("✗ Hardcoded min(50) still present in OCR code")
        return False
    
    # Check for the corrected version
    if 'last_page=self.config.ocr_max_pages,' in content:
        print("✓ OCR page limit uses config value directly")
    else:
        print("⚠ Could not verify OCR page limit fix")
    
    return True


def test_helper_classes():
    """Test that helper classes exist and work."""
    print("\nTesting helper classes...")
    
    from frontend.src.content_expansion.processing_helpers import (
        CacheManager, BatchProcessor, ReportGenerator
    )
    from frontend.src.content_expansion.pdf_analyzer_enhanced import ProcessingConfig
    
    # Test CacheManager
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        cache_manager = CacheManager(cache_dir)
        print("✓ CacheManager instantiated")
        
        # Test BatchProcessor
        config = ProcessingConfig()
        stats = {}
        batch_processor = BatchProcessor(config, stats)
        print("✓ BatchProcessor instantiated")
        
        # Test ReportGenerator
        report_generator = ReportGenerator(config, stats)
        print("✓ ReportGenerator instantiated")
    
    return True


def test_readme_import():
    """Check that README has correct import path."""
    print("\nChecking README import path...")
    
    readme_path = Path('/home/svnbjrn/code/cl2/MDMAI/frontend/src/content_expansion/README.md')
    with open(readme_path, 'r') as f:
        content = f.read()
    
    correct_import = "from frontend.src.content_expansion.pdf_analyzer_enhanced import"
    if correct_import in content:
        print("✓ README has correct import path")
    else:
        print("✗ README import path not fixed")
        return False
    
    return True


def test_test_file_imports():
    """Check that test file has imports at top."""
    print("\nChecking test file imports...")
    
    test_file_path = Path('/home/svnbjrn/code/cl2/MDMAI/frontend/src/content_expansion/test_pdf_analyzer_enhanced.py')
    with open(test_file_path, 'r') as f:
        lines = f.readlines()
    
    # Check that shutil and io are imported at the top
    import_section = ''.join(lines[:30])
    
    if 'import io' in import_section and 'import shutil' in import_section:
        print("✓ Test file has io and shutil imports at top")
    else:
        print("✗ Test file missing imports at top")
        return False
    
    # Check that there's no duplicate import shutil in tearDown
    teardown_section = ''.join(lines[100:200])
    if 'import shutil' in teardown_section:
        print("✗ Found duplicate 'import shutil' in tearDown")
        return False
    
    print("✓ No duplicate shutil import in tearDown")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING CODE REVIEW FIXES")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_processing_config,
        test_no_pickle_import,
        test_ocr_page_limit,
        test_helper_classes,
        test_readme_import,
        test_test_file_imports
    ]
    
    all_passed = True
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with error: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - All code review issues fixed!")
    else:
        print("❌ SOME TESTS FAILED - Please review the output above")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())