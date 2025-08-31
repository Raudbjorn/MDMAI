#!/usr/bin/env python3
"""Test script for Ollama embedding integration."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pdf_processing.ollama_provider import OllamaEmbeddingProvider
from src.pdf_processing.embedding_generator import EmbeddingGenerator
import numpy as np


def test_ollama_provider():
    """Test Ollama provider functionality."""
    print("\n" + "="*60)
    print("TESTING OLLAMA EMBEDDING PROVIDER")
    print("="*60)
    
    # Test 1: Check if Ollama is installed
    provider = OllamaEmbeddingProvider()
    is_installed = provider.check_ollama_installed()
    print(f"\n1. Ollama installed/running: {is_installed}")
    
    if not is_installed:
        print("   Attempting to start Ollama service...")
        started = provider.start_ollama_service()
        print(f"   Service started: {started}")
        if not started:
            print("\n⚠️  Ollama is not installed or cannot be started.")
            print("   Please install Ollama from: https://ollama.ai")
            return False
    
    # Test 2: List available models
    models = provider.list_available_models()
    print(f"\n2. Available models: {models if models else 'None'}")
    
    # Test 3: Test embedding generation with a small model
    test_model = "all-minilm"  # Small, fast model for testing
    print(f"\n3. Testing with model: {test_model}")
    
    if not provider.is_model_available(test_model):
        print(f"   Model not found locally. Downloading {test_model}...")
        success = provider.pull_model(test_model)
        if not success:
            print(f"   Failed to download {test_model}")
            return False
    
    # Test 4: Generate embeddings
    test_provider = OllamaEmbeddingProvider(model_name=test_model)
    test_texts = [
        "The wizard casts a fireball spell.",
        "Roll 3d6 for damage.",
        "The dragon has 200 hit points."
    ]
    
    print(f"\n4. Generating embeddings for {len(test_texts)} texts...")
    try:
        embeddings = test_provider.generate_embeddings_batch(test_texts)
        print(f"   ✓ Generated {len(embeddings)} embeddings")
        print(f"   Embedding dimension: {len(embeddings[0])}")
        
        # Test similarity
        print("\n5. Testing similarity calculation:")
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                vec1 = np.array(embeddings[i])
                vec2 = np.array(embeddings[j])
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                similarities.append((i, j, similarity))
                print(f"   Text {i} vs Text {j}: {similarity:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error generating embeddings: {e}")
        return False


def test_embedding_generator_integration():
    """Test the integrated EmbeddingGenerator with Ollama."""
    print("\n" + "="*60)
    print("TESTING EMBEDDING GENERATOR INTEGRATION")
    print("="*60)
    
    # Test with Ollama
    print("\n1. Testing with Ollama backend:")
    try:
        # Set environment variable to use Ollama
        os.environ["USE_OLLAMA_EMBEDDINGS"] = "true"
        os.environ["OLLAMA_EMBEDDING_MODEL"] = "all-minilm"
        
        generator = EmbeddingGenerator(use_ollama=True, model_name="all-minilm")
        info = generator.get_model_info()
        print(f"   Provider: {info.get('provider')}")
        print(f"   Model: {info.get('model_name')}")
        print(f"   Available: {info.get('is_available')}")
        
        # Generate single embedding
        test_text = "The paladin uses divine smite."
        embedding = generator.generate_single_embedding(test_text)
        print(f"   ✓ Generated embedding with dimension: {len(embedding)}")
        
    except Exception as e:
        print(f"   ✗ Ollama test failed: {e}")
    
    # Test with Sentence Transformers
    print("\n2. Testing with Sentence Transformers backend:")
    try:
        generator_st = EmbeddingGenerator(use_ollama=False)
        info_st = generator_st.get_model_info()
        print(f"   Provider: {info_st.get('provider')}")
        print(f"   Model: {info_st.get('model_name')}")
        print(f"   Device: {info_st.get('device')}")
        print(f"   Dimension: {info_st.get('embedding_dimension')}")
        
        # Generate single embedding
        embedding_st = generator_st.generate_single_embedding(test_text)
        print(f"   ✓ Generated embedding with dimension: {len(embedding_st)}")
        
    except Exception as e:
        print(f"   ✗ Sentence Transformers test failed: {e}")
    
    print("\n" + "="*60)
    print("TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    print("\nStarting Ollama Embedding Tests...")
    print("This will test the Ollama integration for generating embeddings.")
    
    # Run tests
    ollama_ok = test_ollama_provider()
    
    if ollama_ok:
        test_embedding_generator_integration()
    else:
        print("\n⚠️  Skipping integration tests due to Ollama issues.")
        print("   You can still use Sentence Transformers for embeddings.")
    
    print("\nDone!")