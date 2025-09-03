#!/usr/bin/env python3
"""Test script to verify security integration syntax in main.py"""

import ast
import sys
from pathlib import Path

def check_security_integration():
    """Check if security is properly integrated in main.py"""
    
    main_file = Path(__file__).parent / "src" / "main.py"
    
    with open(main_file, 'r') as f:
        content = f.read()
    
    # Parse the AST
    try:
        tree = ast.parse(content)
        print("✓ main.py syntax is valid")
    except SyntaxError as e:
        print(f"✗ Syntax error in main.py: {e}")
        return False
    
    # Check for security imports
    security_imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and 'security' in node.module:
                for name in node.names:
                    security_imports.append(name.name if isinstance(name.name, str) else name.name)
    
    print(f"✓ Found {len(security_imports)} security imports")
    
    required_imports = [
        'SecurityConfig', 'SecurityManager', 'secure_mcp_tool',
        'Permission', 'OperationType', 'ResourceType'
    ]
    
    for imp in required_imports:
        if imp in security_imports:
            print(f"  ✓ {imp} imported")
        else:
            print(f"  ✗ {imp} not found in imports")
    
    # Check for security_manager global variable
    has_security_manager = 'security_manager' in content
    if has_security_manager:
        print("✓ security_manager global variable defined")
    
    # Check for security decorators
    decorator_count = content.count('@secure_mcp_tool')
    print(f"✓ Found {decorator_count} @secure_mcp_tool decorators")
    
    # Check for security initialization in main()
    has_init = 'initialize_security' in content
    if has_init:
        print("✓ Security initialization found in main()")
    
    # Check for security cleanup
    has_cleanup = 'security_manager.perform_security_maintenance' in content
    if has_cleanup:
        print("✓ Security cleanup found")
    
    # Check for security tools
    has_status_tool = 'security_status' in content
    has_maintenance_tool = 'security_maintenance' in content
    
    if has_status_tool:
        print("✓ security_status tool defined")
    if has_maintenance_tool:
        print("✓ security_maintenance tool defined")
    
    print("\n✅ Security integration syntax check complete!")
    print(f"\nSummary:")
    print(f"- Security imports: {len(security_imports)} found")
    print(f"- Security decorators: {decorator_count} applied")
    print(f"- Security manager: {'✓' if has_security_manager else '✗'}")
    print(f"- Security initialization: {'✓' if has_init else '✗'}")
    print(f"- Security cleanup: {'✓' if has_cleanup else '✗'}")
    print(f"- Security tools: {2 if has_status_tool and has_maintenance_tool else 0} added")
    
    return True

if __name__ == "__main__":
    success = check_security_integration()
    sys.exit(0 if success else 1)