#!/usr/bin/env python3
"""
Build script for TTRPG Assistant Desktop Application

Handles building both Python backend and Tauri frontend, then packages everything.
Supports dynamic architecture detection for multiple platforms.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set


class Architecture(StrEnum):
    """Supported architectures."""

    X86_64 = "x86_64"
    AARCH64 = "aarch64"
    ARMV7 = "armv7"
    I686 = "i686"


class OperatingSystem(StrEnum):
    """Supported operating systems."""

    DARWIN = "darwin"
    LINUX = "linux"
    WINDOWS = "windows"


class BuildMode(StrEnum):
    """Build modes."""

    DEBUG = auto()
    RELEASE = auto()


class InstallerTarget(StrEnum):
    """Supported installer targets."""

    MSI = "msi"
    NSIS = "nsis"
    DEB = "deb"
    RPM = "rpm"
    APPIMAGE = "appimage"
    DMG = "dmg"
    ALL = "all"


@dataclass
class PlatformInfo:
    """Platform detection and configuration."""

    os: OperatingSystem
    arch: Architecture
    rust_target: str

    @classmethod
    def detect(cls) -> "PlatformInfo":
        """Detect current platform configuration."""
        os_name = cls._detect_os()
        arch = cls._detect_arch()
        rust_target = cls._build_rust_target(os_name, arch)
        return cls(os=os_name, arch=arch, rust_target=rust_target)

    @staticmethod
    def _detect_os() -> OperatingSystem:
        """Detect operating system using pattern matching."""
        match platform.system().lower():
            case "darwin":
                return OperatingSystem.DARWIN
            case "windows":
                return OperatingSystem.WINDOWS
            case "linux" | _:
                return OperatingSystem.LINUX

    @staticmethod
    def _detect_arch() -> Architecture:
        """Detect system architecture using pattern matching."""
        match platform.machine().lower():
            case "x86_64" | "amd64":
                return Architecture.X86_64
            case "arm64" | "aarch64":
                return Architecture.AARCH64
            case "armv7l" | "armv7":
                return Architecture.ARMV7
            case "i386" | "i686":
                return Architecture.I686
            case unknown:
                print(f"⚠️  Unknown architecture: {unknown}, defaulting to x86_64")
                return Architecture.X86_64

    @staticmethod
    def _build_rust_target(os: OperatingSystem, arch: Architecture) -> str:
        """Build Rust target triple using pattern matching."""
        match os:
            case OperatingSystem.DARWIN:
                return f"{arch}-apple-darwin"
            case OperatingSystem.LINUX:
                # Check for musl vs glibc
                libc = "musl" if PlatformInfo._is_musl_system() else "gnu"
                return f"{arch}-unknown-linux-{libc}"
            case OperatingSystem.WINDOWS:
                return f"{arch}-pc-windows-msvc"
            case _:
                return f"{arch}-unknown-{os.value}"

    @staticmethod
    def _is_musl_system() -> bool:
        """Check if system uses musl libc."""
        try:
            result = subprocess.run(
                ["ldd", "--version"], capture_output=True, text=True, check=False
            )
            return "musl" in result.stdout
        except Exception:
            return False

    @property
    def is_windows(self) -> bool:
        """Check if platform is Windows."""
        return self.os == OperatingSystem.WINDOWS

    @property
    def executable_extension(self) -> str:
        """Get platform-specific executable extension using pattern matching."""
        match self.os:
            case OperatingSystem.WINDOWS:
                return ".exe"
            case _:
                return ""


@dataclass
class BuildConfig:
    """Build configuration settings."""

    root_dir: Path
    platform: PlatformInfo
    mode: BuildMode = BuildMode.RELEASE
    skip_backend: bool = False
    skip_frontend: bool = False
    verbose: bool = False
    installer_targets: Set[InstallerTarget] = field(default_factory=set)
    code_signing: bool = False
    generate_update_manifest: bool = False

    @property
    def is_debug(self) -> bool:
        """Check if building in debug mode."""
        return self.mode == BuildMode.DEBUG

    @property
    def backend_dir(self) -> Path:
        """Get backend directory path."""
        return self.root_dir / "backend"

    @property
    def frontend_dir(self) -> Path:
        """Get frontend directory path."""
        return self.root_dir / "frontend"

    @property
    def tauri_dir(self) -> Path:
        """Get Tauri source directory path."""
        return self.frontend_dir / "src-tauri"

    @property
    def tauri_binaries_dir(self) -> Path:
        """Get Tauri binaries directory path."""
        return self.tauri_dir / "binaries"
    
    @property
    def installer_assets_dir(self) -> Path:
        """Get installer assets directory path."""
        return self.tauri_dir / "installer-assets"
    
    @property
    def update_manifests_dir(self) -> Path:
        """Get update manifests directory path."""
        return self.root_dir / "update-manifests"
    
    def get_platform_targets(self) -> List[InstallerTarget]:
        """Get installer targets supported by current platform."""
        if self.installer_targets and InstallerTarget.ALL not in self.installer_targets:
            return list(self.installer_targets)
            
        match self.platform.os:
            case OperatingSystem.WINDOWS:
                return [InstallerTarget.MSI, InstallerTarget.NSIS]
            case OperatingSystem.DARWIN:
                return [InstallerTarget.DMG]
            case OperatingSystem.LINUX:
                return [InstallerTarget.DEB, InstallerTarget.RPM, InstallerTarget.APPIMAGE]
            case _:
                return [InstallerTarget.APPIMAGE]


@dataclass
class BuildResult:
    """Build result tracking."""

    success: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    artifacts: List[Path] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Add error message."""
        self.errors.append(message)
        self.success = False

    def add_warning(self, message: str) -> None:
        """Add warning message."""
        self.warnings.append(message)

    def add_artifact(self, path: Path) -> None:
        """Add build artifact."""
        if path.exists():
            self.artifacts.append(path)


class DependencyChecker:
    """Check and validate build dependencies."""

    REQUIREMENTS = {
        "python": ["python", "--version"],
        "node": ["node", "--version"],
        "npm": ["npm", "--version"],
        "cargo": ["cargo", "--version"],
        "tauri": ["tauri", "--version"],
    }

    @classmethod
    def check_all(cls) -> BuildResult:
        """Check all required dependencies."""
        result = BuildResult()

        for tool, cmd in cls.REQUIREMENTS.items():
            if not cls._check_tool(tool, cmd):
                result.add_error(f"{tool} is not installed")
            else:
                print(f"✅ {tool} is installed")

        if result.errors:
            cls._print_installation_help(result.errors)

        return result

    @staticmethod
    def _check_tool(name: str, cmd: List[str]) -> bool:
        """Check if a tool is available."""
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    @staticmethod
    def _print_installation_help(missing_tools: List[str]) -> None:
        """Print comprehensive installation instructions for missing tools."""
        print("\n❌ Missing requirements:")
        for tool_error in missing_tools:
            print(f"  - {tool_error}")

        print("\nInstallation instructions:")
        
        # Enhanced installation instructions with platform-specific guidance
        installation_guides = {
            "python": {
                "url": "https://www.python.org/downloads/",
                "instructions": [
                    "Download and install Python 3.11+ from python.org",
                    "Ensure 'python' and 'pip' are in your system PATH",
                    "Verify: python --version && pip --version"
                ]
            },
            "node": {
                "url": "https://nodejs.org/",
                "instructions": [
                    "Download Node.js LTS (18.x+) from nodejs.org",
                    "This includes npm automatically",
                    "Verify: node --version && npm --version"
                ]
            },
            "cargo": {
                "url": "https://rustup.rs/",
                "instructions": [
                    "Install Rust using rustup: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh",
                    "On Windows: Download rustup-init.exe from rustup.rs",
                    "Restart terminal or run: source $HOME/.cargo/env",
                    "Verify: rustc --version && cargo --version"
                ]
            },
            "tauri": {
                "url": "https://tauri.app/v1/guides/getting-started/prerequisites",
                "instructions": [
                    "Prerequisites: Install Rust and Node.js first",
                    "Install Tauri CLI: npm install -g @tauri-apps/cli",
                    "Alternative: cargo install tauri-cli",
                    "Verify: tauri --version",
                    "",
                    "Platform-specific requirements:",
                    "  • Linux: sudo apt install webkit2gtk-4.0-dev build-essential curl wget file libssl-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev",
                    "  • macOS: Xcode Command Line Tools (xcode-select --install)",
                    "  • Windows: Microsoft Visual Studio C++ Build Tools",
                    "",
                    "Troubleshooting:",
                    "  - Permission errors: Use sudo/admin privileges or configure npm prefix",
                    "  - PATH issues: Restart terminal after installation",
                    "  - Rust not found: Ensure ~/.cargo/bin is in PATH"
                ]
            }
        }

        for tool_name, config in installation_guides.items():
            if any(tool_name in error for error in missing_tools):
                print(f"\n{tool_name.upper()} Installation:")
                print(f"  📖 Guide: {config['url']}")
                for instruction in config['instructions']:
                    if instruction:
                        print(f"  • {instruction}")
                    else:
                        print()  # Empty line for spacing


class CommandRunner:
    """Utility for running build commands."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def run(
        self,
        cmd: List[str],
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> int:
        """Run a command with error handling."""
        if description:
            print(f"📦 {description}")
        elif self.verbose:
            print(f"📦 Running: {' '.join(cmd)}")

        if cwd and self.verbose:
            print(f"   in: {cwd}")

        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=not self.verbose,
            text=True,
        )

        if result.returncode != 0:
            print(f"❌ Command failed with code {result.returncode}", file=sys.stderr)
            if not self.verbose and result.stderr:
                print(result.stderr, file=sys.stderr)

        return result.returncode


class BackendBuilder:
    """Python backend builder."""

    def __init__(self, config: BuildConfig, runner: CommandRunner):
        self.config = config
        self.runner = runner

    def build(self) -> BuildResult:
        """Build Python backend."""
        print("\n🐍 Building Python backend...")
        result = BuildResult()

        if self._has_pyinstaller():
            return self._build_with_pyinstaller()
        elif self._has_pyoxidizer():
            return self._build_with_pyoxidizer()
        else:
            return self._create_dev_bundle()

    def _has_pyinstaller(self) -> bool:
        """Check if PyInstaller spec exists."""
        return (self.config.backend_dir / "pyinstaller.spec").exists()

    def _has_pyoxidizer(self) -> bool:
        """Check if PyOxidizer config exists."""
        return (self.config.backend_dir / "pyoxidizer.toml").exists()

    def _build_with_pyinstaller(self) -> BuildResult:
        """Build with PyInstaller."""
        print("Using PyInstaller...")
        result = BuildResult()

        # Install PyInstaller if needed
        self.runner.run(
            [sys.executable, "-m", "pip", "install", "pyinstaller"],
            description="Installing PyInstaller",
        )

        # Build with PyInstaller
        if self.runner.run(
            ["pyinstaller", "pyinstaller.spec"],
            cwd=self.config.backend_dir,
            description="Building with PyInstaller",
        ) != 0:
            result.add_error("PyInstaller build failed")

        # Add artifacts
        dist_dir = self.config.backend_dir / "dist"
        if dist_dir.exists():
            for exe in dist_dir.glob("mcp-server*"):
                result.add_artifact(exe)

        return result

    def _build_with_pyoxidizer(self) -> BuildResult:
        """Build with PyOxidizer."""
        print("Using PyOxidizer...")
        result = BuildResult()

        # Install PyOxidizer if needed
        self.runner.run(
            [sys.executable, "-m", "pip", "install", "pyoxidizer"],
            description="Installing PyOxidizer",
        )

        # Build with PyOxidizer
        if self.runner.run(
            ["pyoxidizer", "build"],
            cwd=self.config.backend_dir,
            description="Building with PyOxidizer",
        ) != 0:
            result.add_error("PyOxidizer build failed")

        # Add artifacts
        build_dir = self.config.backend_dir / "build"
        if build_dir.exists():
            pattern = f"mcp-server{self.config.platform.executable_extension}"
            for exe in build_dir.rglob(pattern):
                result.add_artifact(exe)

        return result

    def _create_dev_bundle(self) -> BuildResult:
        """Create development bundle."""
        print("Creating development bundle...")
        result = BuildResult()

        # Create bundle script
        bundle_script = self.config.backend_dir / "bundle.py"
        bundle_script.write_text(
            """#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.environ['MCP_STDIO_MODE'] = 'true'

from main import main

if __name__ == "__main__":
    main()
"""
        )
        bundle_script.chmod(0o755)

        # Copy Python files
        dist_dir = self.config.backend_dir / "dist"
        dist_dir.mkdir(exist_ok=True)

        for py_file in self.config.backend_dir.glob("*.py"):
            shutil.copy2(py_file, dist_dir)

        result.add_artifact(bundle_script)
        print("  ✓ Created development bundle")
        return result


class InstallerAssetsManager:
    """Manage installer assets and generate missing ones."""
    
    def __init__(self, config: BuildConfig, runner: CommandRunner):
        self.config = config
        self.runner = runner
        
    def check_assets(self) -> BuildResult:
        """Check for required installer assets."""
        result = BuildResult()
        assets_dir = self.config.installer_assets_dir
        
        required_assets = self._get_required_assets()
        missing_assets = []
        
        for asset in required_assets:
            asset_path = assets_dir / asset
            if not asset_path.exists():
                missing_assets.append(asset)
                
        if missing_assets:
            result.add_warning(f"Missing installer assets: {', '.join(missing_assets)}")
            result.add_warning("Placeholder assets will be generated")
            
            # Generate placeholder assets
            self._generate_placeholder_assets(missing_assets)
            
        return result
        
    def _get_required_assets(self) -> List[str]:
        """Get list of required assets for current platform."""
        assets = []
        
        for target in self.config.get_platform_targets():
            match target:
                case InstallerTarget.MSI | InstallerTarget.NSIS:
                    assets.extend(["banner.bmp", "dialog.bmp", "header.bmp", "sidebar.bmp"])
                case InstallerTarget.DMG:
                    assets.append("dmg-background.png")
                    
        return list(set(assets))  # Remove duplicates
        
    def _generate_placeholder_assets(self, missing_assets: List[str]) -> None:
        """Generate placeholder assets for missing files."""
        assets_dir = self.config.installer_assets_dir
        assets_dir.mkdir(parents=True, exist_ok=True)
        
        for asset in missing_assets:
            asset_path = assets_dir / asset
            
            if asset.endswith(".bmp"):
                self._generate_placeholder_bmp(asset_path, asset)
            elif asset.endswith(".png"):
                self._generate_placeholder_png(asset_path, asset)
                
    def _generate_placeholder_bmp(self, path: Path, asset_type: str) -> None:
        """Generate placeholder BMP file."""
        # For now, create a simple text file placeholder
        # In production, you would generate actual BMP files
        with open(path, "w") as f:
            f.write(f"# Placeholder for {asset_type}\n")
            f.write(f"# Replace with actual {asset_type} asset\n")
            f.write(f"# Generated on: {datetime.now().isoformat()}\n")
            
    def _generate_placeholder_png(self, path: Path, asset_type: str) -> None:
        """Generate placeholder PNG file."""
        # For now, create a simple text file placeholder
        # In production, you would generate actual PNG files
        with open(path, "w") as f:
            f.write(f"# Placeholder for {asset_type}\n")
            f.write(f"# Replace with actual {asset_type} asset\n")
            f.write(f"# Generated on: {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}\n")


class CodeSigner:
    """Handle code signing for different platforms."""
    
    def __init__(self, config: BuildConfig, runner: CommandRunner):
        self.config = config
        self.runner = runner
        
    def setup_signing_environment(self) -> BuildResult:
        """Set up code signing environment."""
        result = BuildResult()
        
        if not self.config.code_signing:
            return result
            
        # Check for required environment variables
        required_vars = self._get_required_signing_vars()
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
                
        if missing_vars:
            result.add_error(f"Missing code signing environment variables: {', '.join(missing_vars)}")
            result.add_error("Code signing will be disabled")
            self.config.code_signing = False
            
        return result
        
    def _get_required_signing_vars(self) -> List[str]:
        """Get required environment variables for code signing."""
        base_vars = ["TAURI_SIGNING_PRIVATE_KEY", "TAURI_SIGNING_PRIVATE_KEY_PASSWORD"]
        
        match self.config.platform.os:
            case OperatingSystem.WINDOWS:
                return base_vars + ["WINDOWS_CERTIFICATE_PATH", "WINDOWS_CERTIFICATE_PASSWORD"]
            case OperatingSystem.DARWIN:
                return base_vars + ["MACOS_SIGNING_IDENTITY", "APPLE_ID", "APPLE_PASSWORD", "APPLE_TEAM_ID"]
            case OperatingSystem.LINUX:
                return base_vars + ["GPG_KEY_ID"]
            case _:
                return base_vars
                
    def sign_artifacts(self, artifacts: List[Path]) -> BuildResult:
        """Sign build artifacts."""
        result = BuildResult()
        
        if not self.config.code_signing:
            result.add_warning("Code signing is disabled")
            return result
            
        for artifact in artifacts:
            if self._should_sign_artifact(artifact):
                sign_result = self._sign_single_artifact(artifact)
                result.errors.extend(sign_result.errors)
                result.warnings.extend(sign_result.warnings)
                
        return result
        
    def _should_sign_artifact(self, artifact: Path) -> bool:
        """Check if artifact should be signed."""
        extensions = {".msi", ".exe", ".dmg", ".deb", ".rpm", ".AppImage"}
        return artifact.suffix.lower() in extensions
        
    def _sign_single_artifact(self, artifact: Path) -> BuildResult:
        """Sign a single artifact."""
        result = BuildResult()
        
        # Platform-specific signing logic would go here
        # For now, just log that signing would happen
        print(f"  🔐 Signing {artifact.name}...")
        result.add_warning(f"Code signing for {artifact.name} (placeholder)")
        
        return result


class UpdateManifestGenerator:
    """Generate update manifests for auto-updater."""
    
    def __init__(self, config: BuildConfig, runner: CommandRunner):
        self.config = config
        self.runner = runner
        
    def generate_manifests(self, artifacts: List[Path]) -> BuildResult:
        """Generate update manifests from build artifacts."""
        result = BuildResult()
        
        if not self.config.generate_update_manifest:
            return result
            
        manifest_script = self.config.root_dir / "generate-update-manifest.py"
        if not manifest_script.exists():
            result.add_error("Update manifest generator script not found")
            return result
            
        # Create assets directory and copy artifacts
        assets_dir = self.config.update_manifests_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        
        for artifact in artifacts:
            if self._is_installer_artifact(artifact):
                shutil.copy2(artifact, assets_dir)
                
        # Run manifest generator
        version = self._get_version()
        
        cmd = [
            sys.executable,
            str(manifest_script),
            "--local-assets", str(assets_dir),
            "--version", version,
            "--output-dir", str(self.config.update_manifests_dir),
            "--notes", f"Release {version}"
        ]
        
        if self.runner.run(cmd, description="Generating update manifests") != 0:
            result.add_error("Failed to generate update manifests")
        else:
            # Add generated manifests as artifacts
            for manifest in self.config.update_manifests_dir.glob("*.json"):
                result.add_artifact(manifest)
                
        return result
        
    def _is_installer_artifact(self, artifact: Path) -> bool:
        """Check if artifact is an installer."""
        installer_extensions = {".msi", ".exe", ".dmg", ".deb", ".rpm", ".AppImage"}
        return artifact.suffix.lower() in installer_extensions
        
    def _get_version(self) -> str:
        """Get application version from Tauri config."""
        tauri_config = self.config.tauri_dir / "tauri.conf.json"
        
        if tauri_config.exists():
            try:
                import json
                with open(tauri_config, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                    return config_data.get("version", "1.0.0")
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"⚠️  Warning: Could not read version from {tauri_config}. Error: {e}")
                
        return "1.0.0"


class TauriBuilder:
    """Tauri frontend builder with installer support."""

    def __init__(self, config: BuildConfig, runner: CommandRunner):
        self.config = config
        self.runner = runner
        self.assets_manager = InstallerAssetsManager(config, runner)
        self.code_signer = CodeSigner(config, runner)
        self.manifest_generator = UpdateManifestGenerator(config, runner)

    def prepare_resources(self, backend_artifacts: List[Path]) -> BuildResult:
        """Prepare Tauri resources."""
        print("\n📋 Preparing Tauri resources...")
        result = BuildResult()

        # Create binaries directory
        self.config.tauri_binaries_dir.mkdir(parents=True, exist_ok=True)

        if backend_artifacts:
            # Use compiled backend
            backend_exe = backend_artifacts[0]
            target_name = f"mcp-server-{self.config.platform.rust_target}{self.config.platform.executable_extension}"
            target = self.config.tauri_binaries_dir / target_name
            shutil.copy2(backend_exe, target)
            
            # Create generic symlink for Tauri configuration
            symlink = self.config.tauri_binaries_dir / "mcp-server"
            if symlink.exists():
                symlink.unlink()
            symlink.symlink_to(target_name)  # Create relative symlink
            
            print(f"  ✓ Copied backend to: {target.name}")
            print(f"  ✓ Created symlink: mcp-server -> {target_name}")
            result.add_artifact(target)
        else:
            # Create wrapper for development
            result.add_warning("No compiled backend found, using Python script directly")
            wrapper = self._create_dev_wrapper()
            result.add_artifact(wrapper)

        return result

    def _create_dev_wrapper(self) -> Path:
        """Create development wrapper script."""
        wrapper = self.config.tauri_binaries_dir / "mcp-server.py"
        wrapper.write_text(
            """#!/usr/bin/env python3
import os
import sys

backend_path = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..', 'backend'
)
sys.path.insert(0, backend_path)
os.environ['MCP_STDIO_MODE'] = 'true'

from main import main
main()
"""
        )
        wrapper.chmod(0o755)
        return wrapper

    def build_frontend(self) -> BuildResult:
        """Build frontend assets."""
        print("\n🎨 Building frontend...")
        result = BuildResult()

        # Install dependencies if needed
        if not (self.config.frontend_dir / "node_modules").exists():
            print("Installing frontend dependencies...")
            if self.runner.run(
                ["npm", "install"],
                cwd=self.config.frontend_dir,
                description="Installing dependencies",
            ) != 0:
                result.add_error("Failed to install frontend dependencies")
                return result

        # Build frontend
        if self.runner.run(
            ["npm", "run", "build"],
            cwd=self.config.frontend_dir,
            description="Building frontend assets",
        ) != 0:
            result.add_error("Failed to build frontend")

        return result

    def prepare_installer_assets(self) -> BuildResult:
        """Prepare installer assets."""
        print("\n🎨 Preparing installer assets...")
        return self.assets_manager.check_assets()
        
    def setup_code_signing(self) -> BuildResult:
        """Set up code signing environment."""
        if self.config.code_signing:
            print("\n🔐 Setting up code signing...")
            return self.code_signer.setup_signing_environment()
        return BuildResult()

    def build_app(self) -> BuildResult:
        """Build Tauri application with installer support."""
        print("\n🚀 Building Tauri application...")
        result = BuildResult()

        # Build specific targets if specified
        targets = self.config.get_platform_targets()
        
        for target in targets:
            target_result = self._build_single_target(target)
            result.errors.extend(target_result.errors)
            result.warnings.extend(target_result.warnings)
            result.artifacts.extend(target_result.artifacts)
            
            if not target_result.success:
                result.add_error(f"Failed to build {target.value} installer")
                
        return result
        
    def _build_single_target(self, target: InstallerTarget) -> BuildResult:
        """Build a single installer target."""
        result = BuildResult()
        
        cmd = ["npm", "run", "tauri", "build"]
        
        if self.config.is_debug:
            cmd.append("--debug")
            
        # Add target-specific flags
        cmd.extend(["--target", target.value])
        
        # Add code signing flags if enabled
        if self.config.code_signing:
            cmd.append("--sign")
            
        print(f"  📦 Building {target.value} installer...")
        
        if self.runner.run(
            cmd,
            cwd=self.config.frontend_dir,
            description=f"Building {target.value} installer",
        ) != 0:
            result.add_error(f"Failed to build {target.value} installer")
            return result

        # Find and collect artifacts for this target
        mode = "debug" if self.config.is_debug else "release"
        bundle_dir = self.config.tauri_dir / "target" / mode / "bundle"
        
        if bundle_dir.exists():
            target_artifacts = self._find_target_artifacts(bundle_dir, target)
            
            for artifact in target_artifacts:
                result.add_artifact(artifact)
                print(f"    ✓ Generated: {artifact.name}")
                
        return result
        
    def _find_target_artifacts(self, bundle_dir: Path, target: InstallerTarget) -> List[Path]:
        """Find artifacts for specific target."""
        artifacts = []
        
        patterns = {
            InstallerTarget.MSI: ["**/*.msi"],
            InstallerTarget.NSIS: ["**/*.exe"],
            InstallerTarget.DEB: ["**/*.deb"],
            InstallerTarget.RPM: ["**/*.rpm"],
            InstallerTarget.APPIMAGE: ["**/*.AppImage"],
            InstallerTarget.DMG: ["**/*.dmg"]
        }
        
        for pattern in patterns.get(target, []):
            artifacts.extend(bundle_dir.glob(pattern))
            
        return artifacts
        
    def sign_installers(self, artifacts: List[Path]) -> BuildResult:
        """Sign installer artifacts."""
        if self.config.code_signing:
            print("\n🔐 Signing installers...")
            return self.code_signer.sign_artifacts(artifacts)
        return BuildResult()
        
    def generate_update_manifests(self, artifacts: List[Path]) -> BuildResult:
        """Generate update manifests."""
        if self.config.generate_update_manifest:
            print("\n📋 Generating update manifests...")
            return self.manifest_generator.generate_manifests(artifacts)
        return BuildResult()


class BuildOrchestrator:
    """Orchestrate the entire build process."""

    def __init__(self, config: BuildConfig):
        self.config = config
        self.runner = CommandRunner(config.verbose)
        self.backend_builder = BackendBuilder(config, self.runner)
        self.tauri_builder = TauriBuilder(config, self.runner)

    def build(self) -> BuildResult:
        """Execute complete build process."""
        self._print_build_info()

        # Check dependencies
        dep_result = DependencyChecker.check_all()
        if not dep_result.success:
            return dep_result

        result = BuildResult()

        try:
            # Build backend
            if not self.config.skip_backend:
                backend_result = self.backend_builder.build()
                result.errors.extend(backend_result.errors)
                result.warnings.extend(backend_result.warnings)
                result.artifacts.extend(backend_result.artifacts)

                if not backend_result.success:
                    print("❌ Backend build failed")
                    return result

            # Prepare Tauri resources
            tauri_prep_result = self.tauri_builder.prepare_resources(
                backend_result.artifacts if not self.config.skip_backend else []
            )
            result.warnings.extend(tauri_prep_result.warnings)

            # Build frontend
            if not self.config.skip_frontend:
                frontend_result = self.tauri_builder.build_frontend()
                result.errors.extend(frontend_result.errors)

                if not frontend_result.success:
                    print("❌ Frontend build failed")
                    return result

            # Prepare installer assets
            assets_result = self.tauri_builder.prepare_installer_assets()
            result.warnings.extend(assets_result.warnings)
            
            # Set up code signing
            signing_result = self.tauri_builder.setup_code_signing()
            result.errors.extend(signing_result.errors)
            result.warnings.extend(signing_result.warnings)
            
            if not signing_result.success and self.config.code_signing:
                print("❌ Code signing setup failed")
                return result

            # Build Tauri app with installers
            app_result = self.tauri_builder.build_app()
            result.errors.extend(app_result.errors)
            result.artifacts.extend(app_result.artifacts)

            if not app_result.success:
                print("❌ Tauri build failed")
                return result
                
            # Sign installers if code signing is enabled
            sign_result = self.tauri_builder.sign_installers(app_result.artifacts)
            result.errors.extend(sign_result.errors)
            result.warnings.extend(sign_result.warnings)
            
            # Generate update manifests if requested
            manifest_result = self.tauri_builder.generate_update_manifests(app_result.artifacts)
            result.errors.extend(manifest_result.errors)
            result.warnings.extend(manifest_result.warnings)
            result.artifacts.extend(manifest_result.artifacts)

            self._print_results(result)
            return result

        except KeyboardInterrupt:
            print("\n⚠️  Build interrupted by user")
            result.add_error("Build interrupted")
            return result
        except Exception as e:
            print(f"\n❌ Build failed with error: {e}")
            result.add_error(str(e))
            return result

    def _print_build_info(self) -> None:
        """Print build configuration info."""
        print("🎮 TTRPG Assistant Desktop Build Script")
        print(f"📁 Project root: {self.config.root_dir}")
        print(f"🖥️  Platform: {self.config.platform.os.value}")
        print(f"🏗️  Architecture: {self.config.platform.arch.value}")
        print(f"🎯 Rust Target: {self.config.platform.rust_target}")
        print(f"🔧 Build mode: {self.config.mode.value.capitalize()}")
        print()

    def _print_results(self, result: BuildResult) -> None:
        """Print build results."""
        if result.success:
            print("\n✅ Build completed successfully!")

            if result.artifacts:
                print(f"\n📦 Created {len(result.artifacts)} artifact(s):")
                for artifact in result.artifacts:
                    size_mb = artifact.stat().st_size / (1024 * 1024)
                    print(f"   - {artifact.name} ({size_mb:.1f} MB)")

            if result.warnings:
                print(f"\n⚠️  {len(result.warnings)} warning(s):")
                for warning in result.warnings:
                    print(f"   - {warning}")
        else:
            print("\n❌ Build failed!")
            if result.errors:
                print(f"\nErrors ({len(result.errors)}):")
                for error in result.errors:
                    print(f"   - {error}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build TTRPG Assistant Desktop Application",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Build debug version instead of release",
    )

    parser.add_argument(
        "--skip-backend",
        action="store_true",
        help="Skip backend build step",
    )

    parser.add_argument(
        "--skip-frontend",
        action="store_true",
        help="Skip frontend build step",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--platform",
        choices=["windows", "macos", "linux"],
        help="Target platform (auto-detected by default)",
    )
    
    parser.add_argument(
        "--installer-targets",
        nargs="*",
        choices=[t.value for t in InstallerTarget],
        help="Specific installer targets to build (default: platform-appropriate targets)",
    )
    
    parser.add_argument(
        "--code-signing",
        action="store_true",
        help="Enable code signing for installers",
    )
    
    parser.add_argument(
        "--generate-update-manifest",
        action="store_true",
        help="Generate update manifests for auto-updater",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_arguments()

    # Build configuration
    root_dir = Path(__file__).parent.resolve()
    platform = PlatformInfo.detect()
    mode = BuildMode.DEBUG if args.debug else BuildMode.RELEASE

    # Parse installer targets
    installer_targets = set()
    if args.installer_targets:
        installer_targets = {InstallerTarget(t) for t in args.installer_targets}
    else:
        installer_targets = {InstallerTarget.ALL}
    
    config = BuildConfig(
        root_dir=root_dir,
        platform=platform,
        mode=mode,
        skip_backend=args.skip_backend,
        skip_frontend=args.skip_frontend,
        verbose=args.verbose,
        installer_targets=installer_targets,
        code_signing=args.code_signing,
        generate_update_manifest=args.generate_update_manifest,
    )

    # Execute build
    orchestrator = BuildOrchestrator(config)
    result = orchestrator.build()

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())