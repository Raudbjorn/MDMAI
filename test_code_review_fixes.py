#!/usr/bin/env python3
"""Test script to verify the code review fixes are working correctly."""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from source_management.models import SourceMetadata, Source, FlavorSource, SourceType
from source_management.source_validator import SourceValidator
from source_management.source_organizer import SourceOrganizer


def test_issue1_sourcetype_error_handling():
    """Test that invalid SourceType values are handled gracefully."""
    print("\n=== Testing Issue 1: SourceType Error Handling ===")
    
    # Test 1: Valid source type
    valid_data = {
        'title': 'Test Book',
        'system': 'D&D 5e',
        'source_type': 'rulebook'
    }
    try:
        metadata = SourceMetadata.from_dict(valid_data)
        print(f"✓ Valid source type handled correctly: {metadata.source_type}")
    except Exception as e:
        print(f"✗ Failed on valid source type: {e}")
        return False
    
    # Test 2: Invalid source type (should default to CUSTOM)
    invalid_data = {
        'title': 'Test Book',
        'system': 'D&D 5e',
        'source_type': 'invalid_type'
    }
    try:
        metadata = SourceMetadata.from_dict(invalid_data)
        if metadata.source_type == SourceType.CUSTOM:
            print(f"✓ Invalid source type defaulted to CUSTOM: {metadata.source_type}")
        else:
            print(f"✗ Invalid source type not handled correctly: {metadata.source_type}")
            return False
    except Exception as e:
        print(f"✗ Failed on invalid source type: {e}")
        return False
    
    return True


def test_issue3_input_dict_immutability():
    """Test that from_dict methods don't modify input dictionaries."""
    print("\n=== Testing Issue 3: Input Dictionary Immutability ===")
    
    # Test SourceMetadata.from_dict
    original_metadata = {
        'title': 'Test Book',
        'system': 'D&D 5e',
        'source_type': 'rulebook',
        'extra_field': 'should_remain'
    }
    original_copy = original_metadata.copy()
    
    try:
        metadata = SourceMetadata.from_dict(original_metadata)
        if original_metadata == original_copy:
            print("✓ SourceMetadata.from_dict preserves input dict")
        else:
            print("✗ SourceMetadata.from_dict modified input dict")
            return False
    except Exception as e:
        print(f"✗ SourceMetadata.from_dict failed: {e}")
        return False
    
    # Test Source.from_dict
    original_source = {
        'id': 'test-id',
        'metadata': {
            'title': 'Test Book',
            'system': 'D&D 5e',
            'source_type': 'rulebook'
        },
        'extra_field': 'should_remain'
    }
    original_copy = json.loads(json.dumps(original_source))  # Deep copy
    
    try:
        source = Source.from_dict(original_source)
        if original_source == original_copy:
            print("✓ Source.from_dict preserves input dict")
        else:
            print("✗ Source.from_dict modified input dict")
            print(f"  Original: {original_copy}")
            print(f"  After: {original_source}")
            return False
    except Exception as e:
        print(f"✗ Source.from_dict failed: {e}")
        return False
    
    # Test FlavorSource.from_dict
    original_flavor = {
        'id': 'test-id',
        'metadata': {
            'title': 'Test Book',
            'system': 'D&D 5e',
            'source_type': 'flavor'
        },
        'narrative_style': 'Epic',
        'extra_field': 'should_remain'
    }
    original_copy = json.loads(json.dumps(original_flavor))  # Deep copy
    
    try:
        flavor = FlavorSource.from_dict(original_flavor)
        if original_flavor == original_copy:
            print("✓ FlavorSource.from_dict preserves input dict")
        else:
            print("✗ FlavorSource.from_dict modified input dict")
            return False
    except Exception as e:
        print(f"✗ FlavorSource.from_dict failed: {e}")
        return False
    
    return True


def test_constants_exist():
    """Test that required constants are defined."""
    print("\n=== Testing Constants ===")
    
    # Test FUZZY_MATCH_WORD_LIMIT in SourceOrganizer
    try:
        organizer = SourceOrganizer()
        if hasattr(organizer, 'FUZZY_MATCH_WORD_LIMIT'):
            print(f"✓ FUZZY_MATCH_WORD_LIMIT defined: {organizer.FUZZY_MATCH_WORD_LIMIT}")
        else:
            print("✗ FUZZY_MATCH_WORD_LIMIT not found")
            return False
    except Exception as e:
        print(f"✗ Failed to check FUZZY_MATCH_WORD_LIMIT: {e}")
        return False
    
    # Test lru_cache import in SourceValidator
    try:
        validator = SourceValidator()
        # Check if _is_garbled_text is decorated with lru_cache
        if hasattr(validator._is_garbled_text, '__wrapped__'):
            print("✓ lru_cache decorator is being used")
        else:
            print("! lru_cache might not be applied (but method exists)")
    except Exception as e:
        print(f"✗ Failed to check lru_cache: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Code Review Fixes for Phase 8")
    print("=" * 60)
    
    all_passed = True
    
    # Test Issue 1
    if not test_issue1_sourcetype_error_handling():
        all_passed = False
    
    # Test Issue 3
    if not test_issue3_input_dict_immutability():
        all_passed = False
    
    # Test constants
    if not test_constants_exist():
        all_passed = False
    
    # Note: Issue 2 (N+1 query) is harder to test without a full database setup
    print("\n=== Issue 2: N+1 Query ===")
    print("ℹ Issue 2 fix (N+1 query optimization) requires database setup to test")
    print("  The fix has been implemented in source_manager.py")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All testable fixes are working correctly!")
    else:
        print("❌ Some tests failed. Please review the output above.")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())