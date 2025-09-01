#!/usr/bin/env python3
"""
Generate update manifest for Tauri auto-updater system.
Creates the latest.json file that the updater checks for new versions.
"""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests


@dataclass
class Platform:
    """Platform-specific update information."""
    signature: str
    url: str
    size: int = 0
    checksum: str = ""


@dataclass
class UpdateManifest:
    """Update manifest structure for Tauri updater."""
    version: str
    notes: str
    pub_date: str
    platforms: Dict[str, Platform]


class ManifestGenerator:
    """Generate update manifests for different platforms and release channels."""
    
    PLATFORM_MAPPING = {
        "windows-x86_64": ["windows", "win32", "x64", "msi", "nsis"],
        "windows-i686": ["windows", "win32", "x86", "ia32", "msi", "nsis"],
        "macos-x86_64": ["darwin", "macos", "mac", "x64", "dmg"],
        "macos-aarch64": ["darwin", "macos", "mac", "arm64", "dmg"],
        "linux-x86_64": ["linux", "x64", "deb", "rpm", "appimage"],
        "linux-i686": ["linux", "x86", "ia32", "deb", "rpm", "appimage"],
        "linux-aarch64": ["linux", "arm64", "deb", "rpm", "appimage"],
        "linux-armv7": ["linux", "arm", "armv7", "deb", "rpm", "appimage"],
    }
    
    def __init__(self, github_repo: str = "Raudbjorn/MDMAI", private_key_path: Optional[str] = None):
        self.github_repo = github_repo
        self.private_key_path = private_key_path
        
    def get_github_release(self, tag: str = "latest") -> Dict:
        """Fetch release information from GitHub."""
        url = f"https://api.github.com/repos/{self.github_repo}/releases/{tag}"
        
        # Try to use GitHub token if available
        headers = {}
        if github_token := os.getenv("GITHUB_TOKEN"):
            headers["Authorization"] = f"token {github_token}"
            
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching release data: {e}", file=sys.stderr)
            sys.exit(1)
            
    def calculate_file_hash(self, file_path: Path, algorithm: str = "sha256") -> str:
        """Calculate hash of a file."""
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
                
        return hash_obj.hexdigest()
        
    def sign_file(self, file_path: Path) -> str:
        """Sign file using Tauri's signing mechanism."""
        if not self.private_key_path or not Path(self.private_key_path).exists():
            print(f"Warning: Private key not found at {self.private_key_path}", file=sys.stderr)
            return "unsigned"
        
        try:
            import subprocess
            import hashlib
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import rsa, padding
            
            # Read the file and compute its hash
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            file_hash = hashlib.sha256(file_data).digest()
            
            # Load the private key
            with open(self.private_key_path, 'rb') as key_file:
                private_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=None,  # Assumes no password protection
                )
            
            # Sign the file hash
            if isinstance(private_key, rsa.RSAPrivateKey):
                signature = private_key.sign(
                    file_hash,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                
                # Return base64-encoded signature
                import base64
                return base64.b64encode(signature).decode('ascii')
            else:
                print(f"Warning: Unsupported key type for signing", file=sys.stderr)
                return "unsigned"
                
        except ImportError:
            print(f"Warning: cryptography library not available for signing", file=sys.stderr)
            print("Install with: pip install cryptography", file=sys.stderr)
            return "unsigned"
        except Exception as e:
            print(f"Warning: Failed to sign file {file_path}: {e}", file=sys.stderr)
            return "unsigned"
        
    def match_platform_asset(self, asset_name: str) -> Optional[str]:
        """Match asset filename to platform identifier."""
        asset_lower = asset_name.lower()
        
        for platform, keywords in self.PLATFORM_MAPPING.items():
            if any(keyword in asset_lower for keyword in keywords):
                # Additional checks for specific formats
                if platform.startswith("windows") and (".msi" in asset_lower or ".exe" in asset_lower):
                    return platform
                elif platform.startswith("macos") and ".dmg" in asset_lower:
                    return platform
                elif platform.startswith("linux") and any(ext in asset_lower for ext in [".deb", ".rpm", ".appimage"]):
                    return platform
                    
        return None
        
    def generate_from_github_release(self, tag: str = "latest", output_dir: Path = None) -> Dict[str, UpdateManifest]:
        """Generate update manifest from GitHub release."""
        release_data = self.get_github_release(tag)
        
        version = release_data["tag_name"].lstrip("v")
        notes = release_data.get("body", f"Release {version}")
        pub_date = release_data["published_at"]
        
        # Group assets by platform
        platforms = {}
        
        for asset in release_data.get("assets", []):
            platform = self.match_platform_asset(asset["name"])
            
            if platform:
                # Download asset if needed for signing/checksumming
                asset_path = None
                if output_dir:
                    asset_path = output_dir / asset["name"]
                    if not asset_path.exists():
                        print(f"Downloading {asset['name']}...")
                        self._download_asset(asset["browser_download_url"], asset_path)
                
                # Calculate signature and checksum
                signature = self.sign_file(asset_path) if asset_path else "remote-unsigned"
                checksum = self.calculate_file_hash(asset_path) if asset_path else ""
                
                platforms[platform] = Platform(
                    signature=signature,
                    url=asset["browser_download_url"],
                    size=asset["size"],
                    checksum=checksum
                )
                
        # Generate manifests for different channels
        manifests = {}
        
        # Determine release channel
        if "alpha" in version:
            channel = "alpha"
        elif "beta" in version or "rc" in version:
            channel = "beta" 
        else:
            channel = "stable"
            
        manifests[channel] = UpdateManifest(
            version=version,
            notes=notes,
            pub_date=pub_date,
            platforms=platforms
        )
        
        return manifests
        
    def generate_from_local_files(self, version: str, assets_dir: Path, notes: str = "") -> Dict[str, UpdateManifest]:
        """Generate update manifest from local files."""
        if not assets_dir.exists():
            raise FileNotFoundError(f"Assets directory not found: {assets_dir}")
            
        platforms = {}
        
        for asset_file in assets_dir.glob("*"):
            if asset_file.is_file():
                platform = self.match_platform_asset(asset_file.name)
                
                if platform:
                    signature = self.sign_file(asset_file)
                    checksum = self.calculate_file_hash(asset_file)
                    
                    # For local files, create a placeholder URL
                    url = f"https://github.com/{self.github_repo}/releases/download/v{version}/{asset_file.name}"
                    
                    platforms[platform] = Platform(
                        signature=signature,
                        url=url,
                        size=asset_file.stat().st_size,
                        checksum=checksum
                    )
                    
        # Determine channel from version
        if "alpha" in version:
            channel = "alpha"
        elif "beta" in version or "rc" in version:
            channel = "beta"
        else:
            channel = "stable"
            
        pub_date = datetime.now(timezone.utc).isoformat()
        
        return {
            channel: UpdateManifest(
                version=version,
                notes=notes or f"Release {version}",
                pub_date=pub_date,
                platforms=platforms
            )
        }
        
    def _download_asset(self, url: str, output_path: Path) -> None:
        """Download asset from URL."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
    def save_manifests(self, manifests: Dict[str, UpdateManifest], output_dir: Path) -> None:
        """Save manifests to JSON files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for channel, manifest in manifests.items():
            # Convert to dict and handle Platform objects
            manifest_dict = asdict(manifest)
            
            # Save channel-specific manifest
            channel_file = output_dir / f"latest-{channel}.json"
            with open(channel_file, "w") as f:
                json.dump(manifest_dict, f, indent=2, sort_keys=True)
                
            print(f"Generated {channel_file}")
            
            # For stable channel, also save as latest.json
            if channel == "stable":
                latest_file = output_dir / "latest.json"
                with open(latest_file, "w") as f:
                    json.dump(manifest_dict, f, indent=2, sort_keys=True)
                print(f"Generated {latest_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate Tauri auto-updater manifests",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--repo", 
        default="Raudbjorn/MDMAI",
        help="GitHub repository (owner/name)"
    )
    
    parser.add_argument(
        "--tag",
        default="latest", 
        help="GitHub release tag"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("update-manifests"),
        help="Output directory for manifest files"
    )
    
    parser.add_argument(
        "--private-key",
        type=Path,
        help="Path to private key for signing"
    )
    
    parser.add_argument(
        "--local-assets",
        type=Path,
        help="Use local assets directory instead of GitHub"
    )
    
    parser.add_argument(
        "--version",
        help="Version string (required for local assets)"
    )
    
    parser.add_argument(
        "--notes",
        help="Release notes (for local assets)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.local_assets:
        if not args.version:
            parser.error("--version is required when using --local-assets")
        if not args.local_assets.exists():
            parser.error(f"Local assets directory not found: {args.local_assets}")
            
    # Initialize generator
    generator = ManifestGenerator(
        github_repo=args.repo,
        private_key_path=args.private_key
    )
    
    try:
        # Generate manifests
        if args.local_assets:
            manifests = generator.generate_from_local_files(
                version=args.version,
                assets_dir=args.local_assets,
                notes=args.notes or ""
            )
        else:
            manifests = generator.generate_from_github_release(
                tag=args.tag,
                output_dir=args.output_dir
            )
            
        # Save manifests
        generator.save_manifests(manifests, args.output_dir)
        
        print(f"\nSuccessfully generated {len(manifests)} manifest(s)")
        for channel in manifests:
            print(f"  - {channel}")
            
    except Exception as e:
        print(f"Error generating manifests: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()