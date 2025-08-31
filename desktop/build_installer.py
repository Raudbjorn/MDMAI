#!/usr/bin/env python3
"""
Build script for TTRPG Assistant Desktop Application
Handles building both Python backend and Tauri frontend, then packages everything
"""

import subprocess
import shutil
import sys
import os
from pathlib import Path
from typing import List, Optional


def run_command(cmd: List[str], cwd: Optional[Path] = None, env: Optional[dict] = None) -> int:
    """
    Run a command with error handling
    
    Args:
        cmd: Command and arguments as list
        cwd: Working directory
        env: Environment variables
        
    Returns:
        Return code (0 for success)
    """
    print(f"📦 Running: {' '.join(cmd)}")
    if cwd:
        print(f"   in: {cwd}")
    
    result = subprocess.run(cmd, cwd=cwd, env=env)
    
    if result.returncode != 0:
        print(f"❌ Command failed with code {result.returncode}", file=sys.stderr)
    
    return result.returncode


def check_requirements() -> bool:
    """Check if all required tools are installed"""
    requirements = {
        "python": ["python", "--version"],
        "node": ["node", "--version"],
        "npm": ["npm", "--version"],
        "cargo": ["cargo", "--version"],
        "tauri": ["tauri", "--version"],
    }
    
    missing = []
    
    for tool, cmd in requirements.items():
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"✅ {tool} is installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"❌ {tool} is not installed")
            missing.append(tool)
    
    if missing:
        print(f"\n❌ Missing requirements: {', '.join(missing)}")
        print("\nInstallation instructions:")
        if "python" in missing:
            print("  - Python: https://www.python.org/downloads/")
        if "node" in missing or "npm" in missing:
            print("  - Node.js: https://nodejs.org/")
        if "cargo" in missing:
            print("  - Rust: https://rustup.rs/")
        if "tauri" in missing:
            print("  - Tauri CLI: npm install -g @tauri-apps/cli")
        return False
    
    return True


def build_python_backend(root: Path) -> int:
    """Build Python backend with PyInstaller or PyOxidizer"""
    print("\n🐍 Building Python backend...")
    
    backend_dir = root / "backend"
    
    # Check if we're using PyInstaller or PyOxidizer
    if (backend_dir / "pyinstaller.spec").exists():
        print("Using PyInstaller...")
        
        # Install PyInstaller if needed
        run_command([sys.executable, "-m", "pip", "install", "pyinstaller"])
        
        # Build with PyInstaller
        if run_command(["pyinstaller", "pyinstaller.spec"], cwd=backend_dir) != 0:
            return 1
        
        # The output will be in backend/dist/
        
    elif (backend_dir / "pyoxidizer.toml").exists():
        print("Using PyOxidizer...")
        
        # Install PyOxidizer if needed
        run_command([sys.executable, "-m", "pip", "install", "pyoxidizer"])
        
        # Build with PyOxidizer
        if run_command(["pyoxidizer", "build"], cwd=backend_dir) != 0:
            return 1
        
        # The output will be in backend/build/*/release/
        
    else:
        print("Creating simple Python bundle...")
        
        # Create a simple bundle script
        bundle_script = backend_dir / "bundle.py"
        bundle_script.write_text("""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from main import main

if __name__ == "__main__":
    # Run in stdio mode for Tauri
    import os
    os.environ['MCP_STDIO_MODE'] = 'true'
    main()
""")
        
        # For development, we'll just copy the Python files
        dist_dir = backend_dir / "dist"
        dist_dir.mkdir(exist_ok=True)
        
        for py_file in backend_dir.glob("*.py"):
            shutil.copy2(py_file, dist_dir)
        
        print("  ✓ Created development bundle")
    
    return 0


def prepare_tauri_resources(root: Path) -> int:
    """Copy backend executable to Tauri resources"""
    print("\n📋 Preparing Tauri resources...")
    
    backend_exe = None
    tauri_binaries = root / "frontend" / "src-tauri" / "binaries"
    tauri_binaries.mkdir(parents=True, exist_ok=True)
    
    # Find the backend executable
    possible_paths = [
        root / "backend" / "dist" / "mcp-server.exe",
        root / "backend" / "dist" / "mcp-server",
        root / "backend" / "build" / "x86_64-pc-windows-msvc" / "release" / "mcp-server.exe",
        root / "backend" / "build" / "x86_64-apple-darwin" / "release" / "mcp-server",
        root / "backend" / "build" / "x86_64-unknown-linux-gnu" / "release" / "mcp-server",
    ]
    
    for path in possible_paths:
        if path.exists():
            backend_exe = path
            break
    
    if not backend_exe:
        print("⚠️  No compiled backend found, using Python script directly")
        # For development, create a wrapper script
        wrapper = tauri_binaries / "mcp-server.py"
        wrapper.write_text("""
#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'backend'))
from main import main
# Run in stdio mode for Tauri
import os
os.environ['MCP_STDIO_MODE'] = 'true'
main()
""")
        wrapper.chmod(0o755)
        return 0
    
    # Determine target triple for Tauri
    target_triple = None
    if sys.platform == "win32":
        target_triple = "x86_64-pc-windows-msvc"
        ext = ".exe"
    elif sys.platform == "darwin":
        target_triple = "x86_64-apple-darwin"
        ext = ""
    else:
        target_triple = "x86_64-unknown-linux-gnu"
        ext = ""
    
    # Copy to Tauri binaries with correct naming
    target = tauri_binaries / f"mcp-server-{target_triple}{ext}"
    shutil.copy2(backend_exe, target)
    print(f"  ✓ Copied backend to: {target.name}")
    
    return 0


def build_frontend(root: Path) -> int:
    """Build the Tauri frontend"""
    print("\n🎨 Building frontend...")
    
    frontend_dir = root / "frontend"
    
    # Install dependencies if needed
    if not (frontend_dir / "node_modules").exists():
        print("Installing frontend dependencies...")
        if run_command(["npm", "install"], cwd=frontend_dir) != 0:
            return 1
    
    # Build frontend
    print("Building frontend assets...")
    if run_command(["npm", "run", "build"], cwd=frontend_dir) != 0:
        return 1
    
    return 0


def build_tauri_app(root: Path, release: bool = True) -> int:
    """Build the Tauri application"""
    print("\n🚀 Building Tauri application...")
    
    frontend_dir = root / "frontend"
    
    cmd = ["npm", "run", "tauri", "build"]
    if not release:
        cmd.append("--debug")
    
    if run_command(cmd, cwd=frontend_dir) != 0:
        return 1
    
    # Find output
    bundle_dir = frontend_dir / "src-tauri" / "target" / ("release" if release else "debug") / "bundle"
    
    if bundle_dir.exists():
        print(f"\n✅ Build complete! Installers created in:")
        print(f"   {bundle_dir}")
        
        # List created installers
        installers = list(bundle_dir.glob("**/*.msi")) + \
                    list(bundle_dir.glob("**/*.exe")) + \
                    list(bundle_dir.glob("**/*.dmg")) + \
                    list(bundle_dir.glob("**/*.deb")) + \
                    list(bundle_dir.glob("**/*.AppImage"))
        
        if installers:
            print("\n📦 Created installers:")
            for installer in installers:
                size_mb = installer.stat().st_size / (1024 * 1024)
                print(f"   - {installer.name} ({size_mb:.1f} MB)")
    
    return 0


def main():
    """Main build process"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build TTRPG Assistant Desktop Application")
    parser.add_argument("--debug", action="store_true", help="Build debug version")
    parser.add_argument("--skip-backend", action="store_true", help="Skip backend build")
    parser.add_argument("--skip-frontend", action="store_true", help="Skip frontend build")
    parser.add_argument("--platform", choices=["windows", "macos", "linux"], 
                       help="Target platform (auto-detected by default)")
    
    args = parser.parse_args()
    
    # Find project root
    root = Path(__file__).parent.resolve()
    
    print("🎮 TTRPG Assistant Desktop Build Script")
    print(f"📁 Project root: {root}")
    print(f"🖥️  Platform: {sys.platform}")
    print(f"🔧 Build mode: {'Debug' if args.debug else 'Release'}")
    print()
    
    # Check requirements
    if not check_requirements():
        return 1
    
    # Build steps
    try:
        # 1. Build Python backend
        if not args.skip_backend:
            if build_python_backend(root) != 0:
                print("❌ Backend build failed")
                return 1
        
        # 2. Prepare Tauri resources
        if prepare_tauri_resources(root) != 0:
            print("❌ Resource preparation failed")
            return 1
        
        # 3. Build frontend
        if not args.skip_frontend:
            if build_frontend(root) != 0:
                print("❌ Frontend build failed")
                return 1
        
        # 4. Build Tauri app
        if build_tauri_app(root, release=not args.debug) != 0:
            print("❌ Tauri build failed")
            return 1
        
        print("\n🎉 Build completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Build interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Build failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())