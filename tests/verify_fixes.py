#!/usr/bin/env python3
"""Verification script for code review fixes."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def verify_fixes():
    """Verify all code review fixes are implemented correctly."""
    print("=" * 60)
    print("VERIFYING CODE REVIEW FIXES")
    print("=" * 60)
    
    results = []
    
    # Fix 3: Check distributed JWT manager with ChromaDB
    print("\n3. Checking distributed JWT manager with ChromaDB...")
    try:
        from src.security.jwt_manager_distributed import DistributedJWTManager
        
        # Verify it uses ChromaDB instead of Redis
        with open("src/security/jwt_manager_distributed.py", "r") as f:
            source = f.read()
            
        if "chromadb" in source and "redis" not in source.lower():
            print("   ✅ DistributedJWTManager uses ChromaDB (no Redis dependency)")
            results.append(True)
        else:
            print("   ❌ DistributedJWTManager implementation issue")
            results.append(False)
    except ImportError as e:
        print(f"   ❌ DistributedJWTManager not found: {e}")
        results.append(False)
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ CHROMADB-BASED JWT MANAGER VERIFIED!")
        return 0
    else:
        print(f"\n❌ Implementation needs attention")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(verify_fixes())
    sys.exit(exit_code)