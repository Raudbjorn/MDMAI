"""
Example client showing how to use the port-free IPC bridge
"""

import asyncio
import json
import logging
from typing import Any, Dict, List

from bridge_server import MCPBridge


class TTRPGClient:
    """Example client for TTRPG operations using port-free IPC"""
    
    def __init__(self):
        self.bridge = MCPBridge()
        self.session_id = None
        
    async def connect(self, campaign_id: str = None):
        """Connect to the MCP server"""
        await self.bridge.start()
        self.session_id = await self.bridge.create_session(campaign_id)
        print(f"Connected with session ID: {self.session_id}")
        
    async def disconnect(self):
        """Disconnect from the MCP server"""
        if self.session_id:
            await self.bridge.stop_session(self.session_id)
        await self.bridge.stop()
        print("Disconnected")
    
    async def search_rules(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search for rules in the TTRPG database"""
        result = await self.bridge.call_tool(
            self.session_id,
            "search",
            {
                "query": query,
                "max_results": max_results,
                "use_hybrid": True
            }
        )
        return result.get("results", [])
    
    async def get_campaign_data(self, campaign_id: str) -> Dict[str, Any]:
        """Get campaign data"""
        result = await self.bridge.call_tool(
            self.session_id,
            "get_campaign_data",
            {"campaign_id": campaign_id}
        )
        return result
    
    async def generate_npc(self, system: str, role: str, level: int = None) -> Dict[str, Any]:
        """Generate an NPC"""
        args = {
            "system": system,
            "role": role
        }
        if level:
            args["level"] = level
            
        result = await self.bridge.call_tool(
            self.session_id,
            "generate_npc",
            args
        )
        return result
    
    async def list_available_tools(self) -> List[str]:
        """Get list of available tools"""
        tools = await self.bridge.list_tools(self.session_id)
        return [tool["name"] for tool in tools]


async def demo_basic_operations():
    """Demonstrate basic operations"""
    print("=" * 60)
    print("TTRPG Assistant - Port-free IPC Demo")
    print("=" * 60)
    
    client = TTRPGClient()
    
    try:
        # Connect
        await client.connect(campaign_id="demo_campaign")
        
        # List available tools
        print("\n📋 Available Tools:")
        tools = await client.list_available_tools()
        for tool in tools[:10]:  # Show first 10
            print(f"  - {tool}")
        
        # Search for rules
        print("\n🔍 Searching for 'fireball' spell:")
        results = await client.search_rules("fireball spell", max_results=3)
        for i, result in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"    Content: {result.get('content', '')[:200]}...")
            if 'source' in result:
                print(f"    Source: {result['source']}")
            if 'page' in result:
                print(f"    Page: {result['page']}")
        
        # Generate an NPC
        print("\n🎭 Generating an NPC:")
        npc = await client.generate_npc(
            system="D&D 5e",
            role="merchant",
            level=3
        )
        print(f"  Name: {npc.get('name', 'Unknown')}")
        print(f"  Role: {npc.get('role', 'Unknown')}")
        if 'stats' in npc:
            print(f"  Stats: {json.dumps(npc['stats'], indent=4)}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        
    finally:
        # Disconnect
        await client.disconnect()


async def demo_large_data_handling():
    """Demonstrate handling of large data via Arrow"""
    print("\n" + "=" * 60)
    print("Large Data Handling Demo (via Apache Arrow)")
    print("=" * 60)
    
    client = TTRPGClient()
    
    try:
        await client.connect()
        
        # Simulate a large search that would return many results
        print("\n📚 Performing comprehensive search:")
        results = await client.search_rules(
            "spell",  # Broad search term
            max_results=100  # Large result set
        )
        
        print(f"  Retrieved {len(results)} results")
        print(f"  Data transferred via: {'Apache Arrow (zero-copy)' if len(results) > 20 else 'Protocol Buffers'}")
        
        # Show sample
        if results:
            print(f"\n  Sample result:")
            print(f"    {results[0].get('content', '')[:150]}...")
        
    finally:
        await client.disconnect()


async def demo_performance_comparison():
    """Compare performance of different data transfer methods"""
    import time
    
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)
    
    client = TTRPGClient()
    
    try:
        await client.connect()
        
        # Small data (will use Protocol Buffers)
        print("\n⚡ Small data transfer (Protocol Buffers):")
        start = time.time()
        for _ in range(10):
            await client.search_rules("specific spell name", max_results=1)
        pb_time = time.time() - start
        print(f"  10 requests completed in {pb_time:.3f} seconds")
        print(f"  Average: {pb_time/10*1000:.1f} ms per request")
        
        # Large data (will use Apache Arrow)
        print("\n🚀 Large data transfer (Apache Arrow):")
        start = time.time()
        for _ in range(10):
            await client.search_rules("spell", max_results=50)
        arrow_time = time.time() - start
        print(f"  10 requests completed in {arrow_time:.3f} seconds")
        print(f"  Average: {arrow_time/10*1000:.1f} ms per request")
        
        print(f"\n📊 Performance Summary:")
        print(f"  Protocol Buffers is best for small, frequent messages")
        print(f"  Apache Arrow excels with large tabular data (zero-copy)")
        
    finally:
        await client.disconnect()


async def main():
    """Run all demos"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demos
    await demo_basic_operations()
    await demo_large_data_handling()
    await demo_performance_comparison()
    
    print("\n" + "=" * 60)
    print("✅ All demos completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())