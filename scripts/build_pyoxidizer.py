#!/usr/bin/env python3
"""
Cross-platform PyOxidizer Build Script for MDMAI MCP Server
Builds standalone executables for Linux, Windows, and macOS platforms.

This script automates the PyOxidizer build process, handles platform-specific
configurations, and creates distributable packages for the MCP server.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import platform

class PyOxidizerBuilder:
    """Handles PyOxidizer build operations for MDMAI MCP Server."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.build_dir = project_root / "build" / "pyoxidizer"
        self.dist_dir = project_root / "dist" / "pyoxidizer"
        
        # Platform information
        self.current_platform = platform.system().lower()
        self.current_arch = platform.machine().lower()
        
        # PyOxidizer configuration
        self.config_file = project_root / "pyoxidizer.bzl"
        
    def setup_directories(self):
        """Create necessary build and distribution directories."""
        print("Setting up build directories...")
        
        # Create build directories
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.dist_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Build directory: {self.build_dir}")
        print(f"  Distribution directory: {self.dist_dir}")
    
    def check_pyoxidizer_installation(self) -> bool:
        """Check if PyOxidizer is installed and available."""
        try:
            result = subprocess.run(
                ["pyoxidizer", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            version = result.stdout.strip()
            print(f"PyOxidizer version: {version}")
            return True
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("ERROR: PyOxidizer is not installed or not in PATH")
            print("Install PyOxidizer: https://pyoxidizer.readthedocs.io/en/stable/pyoxidizer_getting_started.html")
            return False
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available."""
        print("Checking dependencies...")
        
        # Check Python version (PyOxidizer works best with Python 3.10+)
        python_version = sys.version_info
        if python_version.major != 3 or python_version.minor < 10:
            print(f"WARNING: Python {python_version.major}.{python_version.minor} detected.")
            print("PyOxidizer works best with Python 3.10 or later.")
        
        # Check if requirements.txt exists
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            print("ERROR: requirements.txt not found in project root")
            return False
        
        # Check if main source files exist
        main_files = [
            self.project_root / "src" / "main.py",
            self.project_root / "src" / "oxidizer_main.py",
        ]
        
        for file_path in main_files:
            if not file_path.exists():
                print(f"ERROR: Required file not found: {file_path}")
                return False
        
        print("All dependencies checked successfully")
        return True
    
    def get_target_triple(self, target_platform: str, target_arch: str) -> str:
        """Get the target triple for the specified platform and architecture."""
        
        # Map platform names
        platform_map = {
            "linux": "unknown-linux-gnu",
            "windows": "pc-windows-msvc", 
            "macos": "apple-darwin",
            "darwin": "apple-darwin",  # Alternative name for macOS
        }
        
        # Map architecture names
        arch_map = {
            "x86_64": "x86_64",
            "amd64": "x86_64", 
            "x64": "x86_64",
            "aarch64": "aarch64",
            "arm64": "aarch64",
            "i386": "i686",
            "i686": "i686",
        }
        
        platform_suffix = platform_map.get(target_platform)
        arch_prefix = arch_map.get(target_arch)
        
        if not platform_suffix or not arch_prefix:
            raise ValueError(f"Unsupported platform/arch combination: {target_platform}/{target_arch}")
        
        return f"{arch_prefix}-{platform_suffix}"
    
    def build_target(self, target: str, platform_override: Optional[str] = None, 
                    arch_override: Optional[str] = None) -> bool:
        """Build a specific target with PyOxidizer."""
        
        print(f"\nBuilding target: {target}")
        
        # Determine target platform and architecture
        target_platform = platform_override or self.current_platform
        target_arch = arch_override or self.current_arch
        
        try:
            # Build command
            cmd = [
                "pyoxidizer", "build",
                "--target-triple", self.get_target_triple(target_platform, target_arch),
                target
            ]
            
            print(f"Running: {' '.join(cmd)}")
            print(f"Working directory: {self.project_root}")
            
            # Run PyOxidizer build
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                check=True,
                capture_output=False,  # Show output in real-time
            )
            
            print(f"✓ Successfully built target: {target}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to build target {target}: {e}")
            return False
        except Exception as e:
            print(f"✗ Unexpected error building target {target}: {e}")
            return False
    
    def package_executable(self, target: str, platform: str, arch: str) -> Optional[Path]:
        """Package the built executable for distribution."""
        
        print(f"Packaging executable for {platform}-{arch}...")
        
        # Determine source and destination paths
        if platform == "windows":
            exe_name = "mdmai-mcp-server.exe"
        else:
            exe_name = "mdmai-mcp-server"
        
        # Find built executable in PyOxidizer output directory
        build_output = self.project_root / "build" / "targets"
        target_triple = self.get_target_triple(platform, arch)
        
        # PyOxidizer typically places executables in:
        # build/targets/{target_triple}/{mode}/{target}/install/
        possible_paths = [
            build_output / target_triple / "release" / target / "install" / exe_name,
            build_output / target_triple / "debug" / target / "install" / exe_name,
            build_output / target_triple / target / exe_name,
        ]
        
        source_exe = None
        for path in possible_paths:
            if path.exists():
                source_exe = path
                break
        
        if not source_exe:
            print(f"ERROR: Could not find built executable for {target}")
            print("Searched in:")
            for path in possible_paths:
                print(f"  - {path}")
            return None
        
        # Create distribution package
        package_name = f"mdmai-mcp-server-{platform}-{arch}"
        package_dir = self.dist_dir / package_name
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy executable
        dest_exe = package_dir / exe_name
        shutil.copy2(source_exe, dest_exe)
        
        # Make executable (Unix systems)
        if platform != "windows":
            dest_exe.chmod(0o755)
        
        # Copy additional files
        additional_files = [
            ("README.md", "README.md"),
            ("LICENSE", "LICENSE"),  # If exists
            ("config", "config"),    # If exists
        ]
        
        for src_name, dest_name in additional_files:
            src_path = self.project_root / src_name
            if src_path.exists():
                dest_path = package_dir / dest_name
                if src_path.is_dir():
                    shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_path, dest_path)
        
        # Create version info file
        version_info = package_dir / "version.txt"
        version_info.write_text(f"""MDMAI MCP Server
Version: 0.1.0
Platform: {platform}
Architecture: {arch}
Build Date: {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}
""")
        
        print(f"✓ Package created: {package_dir}")
        return package_dir
    
    def create_archive(self, package_dir: Path, format: str = "tar.gz") -> Optional[Path]:
        """Create an archive of the packaged executable."""
        
        print(f"Creating {format} archive...")
        
        archive_name = package_dir.name
        
        if format == "tar.gz":
            archive_path = package_dir.parent / f"{archive_name}.tar.gz"
            cmd = ["tar", "-czf", str(archive_path), "-C", str(package_dir.parent), package_dir.name]
        elif format == "zip":
            archive_path = package_dir.parent / f"{archive_name}.zip"
            cmd = ["zip", "-r", str(archive_path), package_dir.name]
        else:
            print(f"ERROR: Unsupported archive format: {format}")
            return None
        
        try:
            subprocess.run(cmd, cwd=package_dir.parent, check=True, capture_output=True)
            print(f"✓ Archive created: {archive_path}")
            return archive_path
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to create archive: {e}")
            return None
    
    def build_all_platforms(self, platforms: Optional[List[str]] = None) -> Dict[str, bool]:
        """Build executables for multiple platforms."""
        
        if platforms is None:
            platforms = ["linux", "windows", "macos"]
        
        # Define platform-specific targets and architectures
        platform_configs = {
            "linux": [("linux-exe", "linux", "x86_64")],
            "windows": [("windows-exe", "windows", "x86_64")],
            "macos": [("macos-exe", "macos", "x86_64"), ("macos-exe", "macos", "aarch64")],
        }
        
        results = {}
        
        for platform in platforms:
            if platform not in platform_configs:
                print(f"WARNING: Unsupported platform: {platform}")
                results[platform] = False
                continue
            
            platform_success = True
            
            for target, plat, arch in platform_configs[platform]:
                print(f"\n{'='*60}")
                print(f"Building {platform} ({arch})")
                print(f"{'='*60}")
                
                # Build the target
                if self.build_target(target, plat, arch):
                    # Package the executable
                    package_dir = self.package_executable(target, plat, arch)
                    if package_dir:
                        # Create archive
                        archive_format = "zip" if plat == "windows" else "tar.gz"
                        self.create_archive(package_dir, archive_format)
                    else:
                        platform_success = False
                else:
                    platform_success = False
            
            results[platform] = platform_success
        
        return results
    
    def clean_build(self):
        """Clean build artifacts."""
        print("Cleaning build artifacts...")
        
        # Remove PyOxidizer build directory
        pyox_build = self.project_root / "build"
        if pyox_build.exists():
            shutil.rmtree(pyox_build)
            print(f"✓ Removed {pyox_build}")
        
        # Remove distribution directory
        if self.dist_dir.exists():
            shutil.rmtree(self.dist_dir)
            print(f"✓ Removed {self.dist_dir}")
        
        print("Build cleanup completed")

def main():
    """Main entry point for the build script."""
    
    parser = argparse.ArgumentParser(
        description="Build MDMAI MCP Server with PyOxidizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/build_pyoxidizer.py --platform linux
  python scripts/build_pyoxidizer.py --platform windows macos
  python scripts/build_pyoxidizer.py --all
  python scripts/build_pyoxidizer.py --clean
        """
    )
    
    parser.add_argument(
        "--platform", "-p",
        nargs="+",
        choices=["linux", "windows", "macos"],
        help="Platform(s) to build for"
    )
    
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Build for all supported platforms"
    )
    
    parser.add_argument(
        "--clean", "-c",
        action="store_true",
        help="Clean build artifacts before building"
    )
    
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Only clean build artifacts, don't build"
    )
    
    args = parser.parse_args()
    
    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Initialize builder
    builder = PyOxidizerBuilder(project_root)
    
    # Clean if requested
    if args.clean or args.clean_only:
        builder.clean_build()
        if args.clean_only:
            return 0
    
    # Set up directories
    builder.setup_directories()
    
    # Check prerequisites
    if not builder.check_pyoxidizer_installation():
        return 1
    
    if not builder.check_dependencies():
        return 1
    
    # Determine platforms to build
    if args.all:
        platforms = ["linux", "windows", "macos"]
    elif args.platform:
        platforms = args.platform
    else:
        # Default to current platform
        current_platform = platform.system().lower()
        if current_platform == "darwin":
            current_platform = "macos"
        platforms = [current_platform]
    
    # Build for specified platforms
    print(f"\nBuilding for platforms: {', '.join(platforms)}")
    results = builder.build_all_platforms(platforms)
    
    # Print results
    print(f"\n{'='*60}")
    print("BUILD SUMMARY")
    print(f"{'='*60}")
    
    success_count = 0
    for platform, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{platform:20} {status}")
        if success:
            success_count += 1
    
    print(f"\nBuilds completed: {success_count}/{len(results)}")
    
    if success_count == len(results):
        print("\n🎉 All builds completed successfully!")
        print(f"\nDistribution packages available in: {builder.dist_dir}")
        return 0
    else:
        print(f"\n❌ {len(results) - success_count} build(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())