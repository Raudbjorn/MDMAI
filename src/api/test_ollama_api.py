"""Test script for Ollama API endpoints."""

import asyncio
import json

import httpx


class APITester:
    """Test suite for Ollama API endpoints."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.tests = [
            ("Health endpoint", self._test_health),
            ("Ollama status", self._test_status),
            ("List models", self._test_models),
            ("Current model", self._test_current),
            ("Model selection", self._test_selection),
            ("Cache clear", self._test_cache),
        ]
    
    async def run_all_tests(self):
        """Run all API tests."""
        print(f"\n{'='*60}\nTESTING OLLAMA API ENDPOINTS\n{'='*60}")
        
        async with httpx.AsyncClient() as client:
            for i, (name, test_func) in enumerate(self.tests, 1):
                print(f"\n{i}. Testing {name}...")
                try:
                    await test_func(client)
                except Exception as e:
                    print(f"   Error: {e}")
        
        print(f"\n{'='*60}\nTESTING COMPLETE\n{'='*60}")
    
    async def _test_health(self, client: httpx.AsyncClient):
        response = await client.get(f"{self.base_url}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    
    async def _test_status(self, client: httpx.AsyncClient):
        response = await client.get(f"{self.base_url}/api/ollama/status")
        data = response.json()
        print(f"   Status: {response.status_code}")
        print(f"   Ollama running: {data.get('is_running', False)}")
        print(f"   Models count: {data.get('models_count', 0)}")
    
    async def _test_models(self, client: httpx.AsyncClient):
        response = await client.get(f"{self.base_url}/api/ollama/models")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            models = response.json()
            print(f"   Found {len(models)} models")
            for model in models[:3]:  # Show first 3
                print(f"     - {model['name']} ({model.get('model_type', 'unknown')})")
    
    async def _test_current(self, client: httpx.AsyncClient):
        response = await client.get(f"{self.base_url}/api/ollama/current")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Current model: {response.json()}")
        else:
            print("   No model selected")
    
    async def _test_selection(self, client: httpx.AsyncClient):
        test_model = "nomic-embed-text"
        print(f"   Attempting to select: {test_model}")
        response = await client.post(
            f"{self.base_url}/api/ollama/select",
            json={"model_name": test_model, "pull_if_missing": False}
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Success: {data.get('success', False)}")
        else:
            print("   Model not available")
    
    async def _test_cache(self, client: httpx.AsyncClient):
        response = await client.delete(f"{self.base_url}/api/ollama/cache")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   {response.json()['message']}")


async def test_ollama_endpoints(base_url: str = "http://localhost:8000"):
    """Test all Ollama API endpoints."""
    tester = APITester(base_url)
    await tester.run_all_tests()


def main():
    """Run the test script."""
    try:
        # Quick health check
        response = httpx.get("http://localhost:8000/health", timeout=2)
        if response.status_code != 200:
            print("API server status unexpected but continuing...")
    except httpx.ConnectError:
        print("\n" + "!" * 60)
        print("ERROR: API server not running!")
        print("Start with: python run_api.py")
        print("!" * 60)
        return
    except Exception as e:
        print(f"Error: {e}")
        return
    
    asyncio.run(test_ollama_endpoints())


if __name__ == "__main__":
    main()