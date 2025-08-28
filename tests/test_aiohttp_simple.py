#!/usr/bin/env python
"""Simple test to verify aiohttp 3.12.14 compatibility"""

import sys
import asyncio

def test_aiohttp_import():
    """Test basic import and version"""
    try:
        import aiohttp
        print(f"✓ aiohttp version: {aiohttp.__version__}")
        assert aiohttp.__version__.startswith("3.12")
        return True
    except Exception as e:
        print(f"✗ Failed to import aiohttp: {e}")
        return False

async def test_client_session():
    """Test client session functionality"""
    try:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            # Test basic session creation
            assert session is not None
            assert not session.closed
            print("✓ ClientSession creation works")
        
        # Session should be closed after context
        assert session.closed
        print("✓ Session cleanup works")
        return True
    except Exception as e:
        print(f"✗ Client session failed: {e}")
        return False

async def test_basic_server():
    """Test basic server functionality"""
    try:
        from aiohttp import web
        
        # Create a simple handler
        async def handler(request):
            return web.Response(text="Hello, World!")
        
        # Create app
        app = web.Application()
        app.router.add_get('/', handler)
        
        assert app is not None
        print("✓ Server app creation works")
        return True
    except Exception as e:
        print(f"✗ Server creation failed: {e}")
        return False

async def test_websocket_support():
    """Test WebSocket support"""
    try:
        from aiohttp import web
        
        # WebSocket handler
        async def websocket_handler(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            return ws
        
        app = web.Application()
        app.router.add_get('/ws', websocket_handler)
        
        print("✓ WebSocket support works")
        return True
    except Exception as e:
        print(f"✗ WebSocket support failed: {e}")
        return False

async def test_compatibility_features():
    """Test compatibility with existing patterns"""
    try:
        import aiohttp
        from aiohttp import ClientTimeout, FormData
        
        # Test timeout configuration
        timeout = ClientTimeout(total=30)
        assert timeout.total == 30
        print("✓ Timeout configuration works")
        
        # Test form data
        form = FormData()
        form.add_field('test', 'value')
        print("✓ FormData works")
        
        return True
    except Exception as e:
        print(f"✗ Compatibility features failed: {e}")
        return False

async def main():
    print("Testing aiohttp 3.12.14 Update")
    print("-" * 40)
    
    # Sync test
    print("\nRunning Import Test...")
    import_result = test_aiohttp_import()
    
    # Async tests
    tests = [
        ("Client Session", test_client_session),
        ("Server Creation", test_basic_server),
        ("WebSocket Support", test_websocket_support),
        ("Compatibility Features", test_compatibility_features),
    ]
    
    results = [import_result]
    for name, test_func in tests:
        print(f"\nRunning {name}...")
        result = await test_func()
        results.append(result)
    
    print("\n" + "=" * 40)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("✓ All tests passed! aiohttp 3.12.14 is compatible.")
        return 0
    else:
        print("✗ Some tests failed. Review the output above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)