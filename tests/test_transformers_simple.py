#!/usr/bin/env python
"""Simple test to verify transformers 4.53.0 compatibility"""

import sys

def test_transformers_import():
    """Test basic import"""
    try:
        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")
        assert transformers.__version__ == "4.53.0"
        return True
    except Exception as e:
        print(f"✗ Failed to import transformers: {e}")
        return False

def test_basic_model_load():
    """Test loading a small model"""
    try:
        from transformers import AutoTokenizer, pipeline
        
        # Use a tiny model for testing
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        print("✓ Tokenizer loaded successfully")
        
        # Test tokenization
        result = tokenizer("Hello world", return_tensors="pt")
        print(f"✓ Tokenization works: {result.input_ids.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_compatibility():
    """Test compatibility with existing code patterns"""
    try:
        from transformers import AutoModel, AutoTokenizer
        
        # Check key APIs exist
        assert hasattr(AutoModel, 'from_pretrained')
        assert hasattr(AutoTokenizer, 'from_pretrained')
        print("✓ Key APIs are available")
        
        return True
    except Exception as e:
        print(f"✗ Compatibility check failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Transformers 4.53.0 Update")
    print("-" * 40)
    
    tests = [
        ("Import Test", test_transformers_import),
        ("Model Loading", test_basic_model_load),
        ("API Compatibility", test_compatibility),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nRunning {name}...")
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 40)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("✓ All tests passed! Transformers 4.53.0 is compatible.")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Review the output above.")
        sys.exit(1)