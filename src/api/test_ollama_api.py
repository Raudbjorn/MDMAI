"""Test script for Ollama API endpoints."""

import asyncio
import json
from typing import Any, Dict

import httpx


async def test_ollama_endpoints(base_url: str = "http://localhost:8000"):
    """
    Test all Ollama API endpoints.
    
    Args:
        base_url: Base URL of the API server
    """
    async with httpx.AsyncClient() as client:
        print("\n" + "="*60)
        print("TESTING OLLAMA API ENDPOINTS")
        print("="*60)
        
        # Test health endpoint
        print("\n1. Testing health endpoint...")
        try:
            response = await client.get(f"{base_url}/health")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {json.dumps(response.json(), indent=2)}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test Ollama status
        print("\n2. Testing Ollama status...")
        try:
            response = await client.get(f"{base_url}/api/ollama/status")
            print(f"   Status: {response.status_code}")
            data = response.json()
            print(f"   Ollama running: {data.get('is_running', False)}")
            print(f"   Models count: {data.get('models_count', 0)}")
            print(f"   API URL: {data.get('api_url', 'N/A')}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test list models
        print("\n3. Testing list models...")
        try:
            response = await client.get(f"{base_url}/api/ollama/models")
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                models = response.json()
                print(f"   Found {len(models)} models:")
                for model in models[:5]:  # Show first 5 models
                    print(f"     - {model['name']} ({model['model_type']})")
                    if model.get('dimension'):
                        print(f"       Dimension: {model['dimension']}")
                if len(models) > 5:
                    print(f"     ... and {len(models) - 5} more")
            else:
                print(f"   Error response: {response.text}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test current model
        print("\n4. Testing get current model...")
        try:
            response = await client.get(f"{base_url}/api/ollama/current")
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                print(f"   Current model: {json.dumps(response.json(), indent=2)}")
            else:
                print(f"   No model selected (expected for fresh start)")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test model selection
        print("\n5. Testing model selection...")
        try:
            # Try to select a common embedding model
            test_model = "nomic-embed-text"
            print(f"   Attempting to select model: {test_model}")
            response = await client.post(
                f"{base_url}/api/ollama/select",
                json={
                    "model_name": test_model,
                    "pull_if_missing": False
                }
            )
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"   Success: {data.get('success', False)}")
                print(f"   Message: {data.get('message', 'N/A')}")
                if data.get('dimension'):
                    print(f"   Dimension: {data['dimension']}")
            else:
                print(f"   Model not available (expected if not installed)")
                error_data = response.json()
                if "details" in error_data and "available_models" in error_data["details"]:
                    available = error_data["details"]["available_models"]
                    if available:
                        print(f"   Available models: {', '.join(available[:3])}")
        except Exception as e:
            print(f"   Error: {e}")
        
        # Test cache clearing
        print("\n6. Testing cache clear...")
        try:
            response = await client.delete(f"{base_url}/api/ollama/cache")
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                print(f"   Response: {response.json()}")
        except Exception as e:
            print(f"   Error: {e}")
        
        print("\n" + "="*60)
        print("TESTING COMPLETE")
        print("="*60)


def main():
    """Run the test script."""
    # Check if API server is running
    import httpx
    
    try:
        response = httpx.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("API server is running, proceeding with tests...")
        else:
            print("API server returned unexpected status, but continuing...")
    except httpx.ConnectError:
        print("\n" + "!"*60)
        print("WARNING: API server is not running!")
        print("Please start the API server first with:")
        print("  python -m src.api.main")
        print("or:")
        print("  uvicorn src.api.main:app --reload")
        print("!"*60 + "\n")
        return
    except Exception as e:
        print(f"Error checking API server: {e}")
        return
    
    # Run async tests
    asyncio.run(test_ollama_endpoints())


if __name__ == "__main__":
    main()