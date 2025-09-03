#!/usr/bin/env python
"""Simple test to verify torch 2.8.0 compatibility"""

import sys

def test_torch_import():
    """Test basic import and version"""
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        assert torch.__version__.startswith("2.8")
        return True
    except Exception as e:
        print(f"✗ Failed to import torch: {e}")
        return False

def test_basic_operations():
    """Test basic tensor operations"""
    try:
        import torch
        
        # Create tensors
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])
        
        # Basic operations
        z = x + y
        assert z.tolist() == [5.0, 7.0, 9.0]
        print("✓ Basic tensor operations work")
        
        # Autograd
        x = torch.tensor([1.0], requires_grad=True)
        y = x * 2
        y.backward()
        assert x.grad.item() == 2.0
        print("✓ Autograd works")
        
        return True
    except Exception as e:
        print(f"✗ Tensor operations failed: {e}")
        return False

def test_nn_modules():
    """Test neural network modules"""
    try:
        import torch
        import torch.nn as nn
        
        # Simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
        
        # Forward pass
        x = torch.randn(1, 10)
        output = model(x)
        assert output.shape == torch.Size([1, 2])
        print("✓ Neural network modules work")
        
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters())
        loss = output.sum()
        loss.backward()
        optimizer.step()
        print("✓ Optimizer works")
        
        return True
    except Exception as e:
        print(f"✗ NN modules failed: {e}")
        return False

def test_compatibility_with_transformers():
    """Test compatibility with transformers library"""
    try:
        import torch
        import transformers
        
        # Check if transformers can use torch
        from transformers import AutoModel
        
        # This should not throw an error
        assert hasattr(transformers, '__version__')
        assert hasattr(torch, '__version__')
        print(f"✓ Compatible with transformers {transformers.__version__}")
        
        return True
    except Exception as e:
        print(f"✗ Compatibility check failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing PyTorch 2.8.0 Update")
    print("-" * 40)
    
    tests = [
        ("Import Test", test_torch_import),
        ("Basic Operations", test_basic_operations),
        ("NN Modules", test_nn_modules),
        ("Transformers Compatibility", test_compatibility_with_transformers),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nRunning {name}...")
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 40)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("✓ All tests passed! PyTorch 2.8.0 is compatible.")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Review the output above.")
        sys.exit(1)