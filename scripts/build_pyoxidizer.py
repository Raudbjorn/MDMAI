#!/usr/bin/env python3
"""
Cross-platform PyOxidizer Build Automation for MDMAI MCP Server.

This module provides a comprehensive, production-ready build system for creating
standalone PyOxidizer executables across multiple platforms. It features advanced
error handling, performance optimization, and extensive configuration management.

Features:
    - Multi-platform build support (Linux, Windows, macOS)
    - Comprehensive dependency validation
    - Advanced build configuration management
    - Performance monitoring and optimization
    - Robust error handling with detailed diagnostics
    - Automated packaging and distribution
    - Parallel build capabilities
    - Type-safe implementation with full documentation
"""

import argparse
import asyncio
import concurrent.futures
import functools
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


class BuildError(Exception):
    """Base exception for build-related errors."""
    pass


class DependencyError(BuildError):
    """Raised when build dependencies are missing or invalid."""
    pass


class BuildTargetError(BuildError):
    """Raised when build target configuration is invalid."""
    pass


class Platform(Enum):
    """Supported build platforms."""
    LINUX = "linux"
    WINDOWS = "windows" 
    MACOS = "macos"
    DARWIN = "darwin"  # Alias for macOS


class Architecture(Enum):
    """Supported architectures."""
    X86_64 = "x86_64"
    AARCH64 = "aarch64"
    ARM64 = "arm64"
    I686 = "i686"


@dataclass
class BuildTarget:
    """Configuration for a specific build target."""
    name: str
    platform: Platform
    architecture: Architecture
    target_triple: str
    executable_name: str
    
    def __post_init__(self) -> None:
        """Validate target configuration after initialization."""
        if not self.name or not self.target_triple:
            raise BuildTargetError(f"Invalid build target configuration: {self}")


@dataclass
class BuildResult:
    """Result of a build operation."""
    target: BuildTarget
    success: bool
    build_time: float
    error_message: Optional[str] = None
    executable_path: Optional[Path] = None
    package_path: Optional[Path] = None


@dataclass
class BuildConfiguration:
    """Comprehensive build configuration."""
    project_root: Path
    build_dir: Path = field(init=False)
    dist_dir: Path = field(init=False)
    
    # Build settings
    clean_build: bool = False
    parallel_builds: int = field(default_factory=lambda: min(4, os.cpu_count() or 4))
    build_timeout: int = 1800  # 30 minutes
    
    # Target configuration
    supported_targets: Dict[str, List[BuildTarget]] = field(init=False)
    
    def __post_init__(self) -> None:
        """Initialize computed fields after dataclass creation."""
        self.build_dir = self.project_root / "build" / "pyoxidizer"
        self.dist_dir = self.project_root / "dist" / "pyoxidizer"
        self._initialize_supported_targets()
    
    def _initialize_supported_targets(self) -> None:
        """Initialize supported build targets."""
        self.supported_targets = {
            Platform.LINUX.value: [
                BuildTarget(
                    name="linux-exe",
                    platform=Platform.LINUX,
                    architecture=Architecture.X86_64,
                    target_triple="x86_64-unknown-linux-gnu",
                    executable_name="mdmai-mcp-server"
                )
            ],
            Platform.WINDOWS.value: [
                BuildTarget(
                    name="windows-exe", 
                    platform=Platform.WINDOWS,
                    architecture=Architecture.X86_64,
                    target_triple="x86_64-pc-windows-msvc",
                    executable_name="mdmai-mcp-server.exe"
                )
            ],
            Platform.MACOS.value: [
                BuildTarget(
                    name="macos-exe",
                    platform=Platform.MACOS,
                    architecture=Architecture.X86_64,
                    target_triple="x86_64-apple-darwin",
                    executable_name="mdmai-mcp-server"
                ),
                BuildTarget(
                    name="macos-exe",
                    platform=Platform.MACOS,
                    architecture=Architecture.AARCH64,
                    target_triple="aarch64-apple-darwin",
                    executable_name="mdmai-mcp-server"
                )
            ]
        }


class DependencyValidator:
    """Validates build dependencies and environment."""
    
    REQUIRED_TOOLS = ["pyoxidizer"]
    REQUIRED_FILES = ["requirements.txt", "src/main.py", "src/oxidizer_main.py"]
    MIN_PYTHON_VERSION = (3, 10)
    
    def __init__(self, project_root: Path) -> None:
        """Initialize dependency validator.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
    
    def validate_all(self) -> None:
        """Validate all dependencies and environment requirements.
        
        Raises:
            DependencyError: If any validation fails
        """
        validation_steps = [
            ("Python version", self._validate_python_version),
            ("Required tools", self._validate_tools),
            ("Project files", self._validate_files),
            ("PyOxidizer configuration", self._validate_pyoxidizer_config),
        ]
        
        for step_name, validator in validation_steps:
            try:
                logger.info(f"Validating: {step_name}")
                validator()
                logger.info(f"✓ {step_name} validated")
            except Exception as e:
                error_msg = f"{step_name} validation failed: {e}"
                logger.error(f"✗ {error_msg}")
                raise DependencyError(error_msg) from e
    
    def _validate_python_version(self) -> None:
        """Validate Python version meets minimum requirements."""
        current_version = sys.version_info[:2]
        if current_version < self.MIN_PYTHON_VERSION:
            raise DependencyError(
                f"Python {'.'.join(map(str, self.MIN_PYTHON_VERSION))}+ required, "
                f"found {'.'.join(map(str, current_version))}"
            )
        logger.info(f"Python version: {sys.version}")
    
    def _validate_tools(self) -> None:
        """Validate required build tools are available."""
        for tool in self.REQUIRED_TOOLS:
            if not self._check_tool_availability(tool):
                raise DependencyError(f"Required tool not found: {tool}")
    
    def _check_tool_availability(self, tool: str) -> bool:
        """Check if a build tool is available in PATH."""
        try:
            result = subprocess.run(
                [tool, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info(f"{tool} version: {result.stdout.strip()}")
                return True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        return False
    
    def _validate_files(self) -> None:
        """Validate required project files exist."""
        missing_files = []
        for file_path in self.REQUIRED_FILES:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(str(full_path))
        
        if missing_files:
            raise DependencyError(f"Required files not found: {missing_files}")
    
    def _validate_pyoxidizer_config(self) -> None:
        """Validate PyOxidizer configuration file."""
        config_file = self.project_root / "pyoxidizer.bzl"
        if not config_file.exists():
            logger.warning(f"PyOxidizer config not found: {config_file}")


class BuildExecutor:
    """Executes PyOxidizer build operations with advanced error handling."""
    
    def __init__(self, config: BuildConfiguration) -> None:
        """Initialize build executor.
        
        Args:
            config: Build configuration
        """
        self.config = config
        self.dependency_validator = DependencyValidator(config.project_root)
    
    def prepare_build_environment(self) -> None:
        """Prepare the build environment."""
        logger.info("Preparing build environment")
        
        # Validate dependencies
        self.dependency_validator.validate_all()
        
        # Create directories
        self._create_build_directories()
        
        # Clean if requested
        if self.config.clean_build:
            self._clean_build_artifacts()
    
    def _create_build_directories(self) -> None:
        """Create necessary build directories."""
        directories = [self.config.build_dir, self.config.dist_dir]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
    
    def _clean_build_artifacts(self) -> None:
        """Clean existing build artifacts."""
        logger.info("Cleaning build artifacts")
        
        artifacts_to_clean = [
            self.config.project_root / "build",
            self.config.dist_dir
        ]
        
        for artifact in artifacts_to_clean:
            if artifact.exists():
                shutil.rmtree(artifact)
                logger.info(f"Cleaned: {artifact}")
    
    def build_target(self, target: BuildTarget) -> BuildResult:
        """Build a specific target.
        
        Args:
            target: Target to build
            
        Returns:
            BuildResult with build outcome
        """
        start_time = time.time()
        logger.info(f"Building target: {target.name} ({target.platform.value}-{target.architecture.value})")
        
        try:
            # Execute build command
            self._execute_build_command(target)
            
            # Find and validate executable
            executable_path = self._locate_built_executable(target)
            if not executable_path:
                raise BuildError(f"Built executable not found for target: {target.name}")
            
            # Package executable
            package_path = self._package_executable(target, executable_path)
            
            build_time = time.time() - start_time
            logger.info(f"✓ Successfully built {target.name} in {build_time:.2f}s")
            
            return BuildResult(
                target=target,
                success=True,
                build_time=build_time,
                executable_path=executable_path,
                package_path=package_path
            )
            
        except Exception as e:
            build_time = time.time() - start_time
            error_msg = f"Build failed for {target.name}: {e}"
            logger.error(f"✗ {error_msg}")
            
            return BuildResult(
                target=target,
                success=False,
                build_time=build_time,
                error_message=error_msg
            )
    
    def _execute_build_command(self, target: BuildTarget) -> None:
        """Execute PyOxidizer build command for target."""
        cmd = [
            "pyoxidizer", "build",
            "--target-triple", target.target_triple,
            target.name
        ]
        
        logger.debug(f"Executing: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.config.project_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.config.build_timeout
            )
            
            # Log build output for debugging
            if result.stdout:
                logger.debug(f"Build stdout: {result.stdout}")
            if result.stderr:
                logger.debug(f"Build stderr: {result.stderr}")
                
        except subprocess.TimeoutExpired as e:
            raise BuildError(f"Build timed out after {self.config.build_timeout}s") from e
        except subprocess.CalledProcessError as e:
            raise BuildError(f"Build command failed: {e.stderr}") from e
    
    def _locate_built_executable(self, target: BuildTarget) -> Optional[Path]:
        """Locate the built executable for a target."""
        build_output = self.config.project_root / "build" / "targets"
        
        # Common paths where PyOxidizer places executables
        search_paths = [
            build_output / target.target_triple / "release" / target.name / "install" / target.executable_name,
            build_output / target.target_triple / "debug" / target.name / "install" / target.executable_name,
            build_output / target.target_triple / target.name / target.executable_name,
        ]
        
        for path in search_paths:
            if path.exists() and path.is_file():
                logger.debug(f"Found executable: {path}")
                return path
        
        # Log search paths for debugging
        logger.error(f"Executable not found in search paths:")
        for path in search_paths:
            logger.error(f"  - {path}")
        
        return None
    
    def _package_executable(self, target: BuildTarget, executable_path: Path) -> Path:
        """Package the built executable for distribution."""
        package_name = f"mdmai-mcp-server-{target.platform.value}-{target.architecture.value}"
        package_dir = self.config.dist_dir / package_name
        
        # Create package directory
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy executable
        dest_executable = package_dir / target.executable_name
        shutil.copy2(executable_path, dest_executable)
        
        # Set executable permissions (Unix systems)
        if target.platform != Platform.WINDOWS:
            dest_executable.chmod(0o755)
        
        # Copy additional files
        self._copy_additional_files(package_dir)
        
        # Create version info
        self._create_version_info(package_dir, target)
        
        logger.info(f"✓ Package created: {package_dir}")
        return package_dir
    
    def _copy_additional_files(self, package_dir: Path) -> None:
        """Copy additional files to package directory."""
        additional_files = [
            ("README.md", "README.md"),
            ("LICENSE", "LICENSE"),
            ("config", "config")
        ]
        
        for src_name, dest_name in additional_files:
            src_path = self.config.project_root / src_name
            if src_path.exists():
                dest_path = package_dir / dest_name
                
                if src_path.is_dir():
                    shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_path, dest_path)
                
                logger.debug(f"Copied {src_name} to package")
    
    def _create_version_info(self, package_dir: Path, target: BuildTarget) -> None:
        """Create version information file."""
        version_info = package_dir / "version.txt"
        
        # Get current timestamp
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        except ImportError:
            timestamp = "unknown"
        
        version_content = f"""MDMAI MCP Server
Version: 0.1.0
Platform: {target.platform.value}
Architecture: {target.architecture.value}
Target Triple: {target.target_triple}
Build Date: {timestamp}
Python Version: {sys.version}
"""
        
        version_info.write_text(version_content)
        logger.debug("Created version info file")
    
    def build_multiple_targets(self, targets: List[BuildTarget]) -> List[BuildResult]:
        """Build multiple targets with optional parallelization."""
        if len(targets) <= 1:
            return [self.build_target(targets[0])] if targets else []
        
        logger.info(f"Building {len(targets)} targets with {self.config.parallel_builds} parallel workers")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.parallel_builds) as executor:
            # Submit all build tasks
            future_to_target = {
                executor.submit(self.build_target, target): target 
                for target in targets
            }
            
            # Collect results
            results = []
            for future in concurrent.futures.as_completed(future_to_target):
                target = future_to_target[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Unexpected error building {target.name}: {e}")
                    results.append(BuildResult(
                        target=target,
                        success=False,
                        build_time=0.0,
                        error_message=str(e)
                    ))
            
            return results


class ArchiveCreator:
    """Creates distribution archives from packaged executables."""
    
    @staticmethod
    def create_archive(package_dir: Path, format: str = "auto") -> Optional[Path]:
        """Create an archive from a package directory.
        
        Args:
            package_dir: Directory to archive
            format: Archive format ('tar.gz', 'zip', or 'auto')
            
        Returns:
            Path to created archive, or None if creation failed
        """
        if format == "auto":
            # Auto-select format based on platform
            format = "zip" if platform.system().lower() == "windows" else "tar.gz"
        
        logger.info(f"Creating {format} archive for {package_dir.name}")
        
        archive_name = package_dir.name
        
        if format == "tar.gz":
            archive_path = package_dir.parent / f"{archive_name}.tar.gz"
            cmd = ["tar", "-czf", str(archive_path), "-C", str(package_dir.parent), package_dir.name]
        elif format == "zip":
            archive_path = package_dir.parent / f"{archive_name}.zip"
            cmd = ["zip", "-r", str(archive_path), package_dir.name]
        else:
            logger.error(f"Unsupported archive format: {format}")
            return None
        
        try:
            subprocess.run(cmd, cwd=package_dir.parent, check=True, capture_output=True)
            logger.info(f"✓ Archive created: {archive_path}")
            return archive_path
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to create archive: {e}")
            return None


class PyOxidizerBuilder:
    """Main builder class orchestrating the entire build process."""
    
    def __init__(self, project_root: Path) -> None:
        """Initialize the PyOxidizer builder.
        
        Args:
            project_root: Root directory of the project
        """
        self.config = BuildConfiguration(project_root=project_root)
        self.executor = BuildExecutor(self.config)
        self.archive_creator = ArchiveCreator()
    
    def build_platforms(self, platforms: Optional[List[str]] = None) -> Dict[str, bool]:
        """Build executables for specified platforms.
        
        Args:
            platforms: List of platform names to build, or None for current platform
            
        Returns:
            Dictionary mapping platform names to build success status
        """
        if platforms is None:
            current_platform = platform.system().lower()
            if current_platform == "darwin":
                current_platform = "macos"
            platforms = [current_platform]
        
        logger.info(f"Building for platforms: {platforms}")
        
        # Prepare build environment
        self.executor.prepare_build_environment()
        
        # Collect all targets
        all_targets = []
        for platform_name in platforms:
            if platform_name in self.config.supported_targets:
                all_targets.extend(self.config.supported_targets[platform_name])
            else:
                logger.warning(f"Unsupported platform: {platform_name}")
        
        if not all_targets:
            logger.error("No valid targets found for specified platforms")
            return {platform_name: False for platform_name in platforms}
        
        # Build all targets
        results = self.executor.build_multiple_targets(all_targets)
        
        # Create archives for successful builds
        for result in results:
            if result.success and result.package_path:
                self.archive_creator.create_archive(result.package_path)
        
        # Generate summary
        platform_results = {}
        for platform_name in platforms:
            platform_targets = [r for r in results if r.target.platform.value == platform_name]
            platform_results[platform_name] = all(r.success for r in platform_targets)
        
        return platform_results
    
    def clean_build_artifacts(self) -> None:
        """Clean all build artifacts."""
        self.executor._clean_build_artifacts()


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Build MDMAI MCP Server with PyOxidizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --platform linux
  %(prog)s --platform windows macos
  %(prog)s --all
  %(prog)s --clean
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
    
    parser.add_argument(
        "--parallel-builds",
        type=int,
        default=min(4, os.cpu_count() or 4),
        help="Number of parallel build workers"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser


def main() -> int:
    """Main entry point for the build script.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    try:
        # Initialize builder
        builder = PyOxidizerBuilder(project_root)
        builder.config.clean_build = args.clean
        builder.config.parallel_builds = args.parallel_builds
        
        # Handle clean-only mode
        if args.clean_only:
            builder.clean_build_artifacts()
            logger.info("Build artifacts cleaned")
            return 0
        
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
        
        # Execute builds
        logger.info("=" * 60)
        logger.info("MDMAI MCP Server - PyOxidizer Build")
        logger.info("=" * 60)
        
        results = builder.build_platforms(platforms)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("BUILD SUMMARY")
        logger.info("=" * 60)
        
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        
        for platform_name, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            logger.info(f"{platform_name:20} {status}")
        
        logger.info(f"\nBuilds completed: {success_count}/{total_count}")
        
        if success_count == total_count:
            logger.info("🎉 All builds completed successfully!")
            logger.info(f"Distribution packages available in: {builder.config.dist_dir}")
            return 0
        else:
            logger.error(f"❌ {total_count - success_count} build(s) failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Build interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        if args.verbose:
            logger.error("Full traceback:", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())