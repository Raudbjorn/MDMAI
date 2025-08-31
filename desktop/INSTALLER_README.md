# TTRPG Assistant - Platform-Specific Installers

This document provides comprehensive information about building, configuring, and deploying platform-specific installers for the TTRPG Assistant desktop application.

## Overview

The TTRPG Assistant uses Tauri 2.1 to create native installers for Windows, macOS, and Linux platforms. The installer system includes:

- **Platform-specific packages**: MSI/NSIS (Windows), DMG (macOS), DEB/RPM/AppImage (Linux)
- **Code signing**: Full certificate-based signing for all platforms
- **Auto-updater**: Seamless background updates with signed manifests
- **Asset management**: Professional installer branding and assets

## Quick Start

### Building All Installers
```bash
# Build all platform-appropriate installers
./build.sh installers

# Build with code signing (requires certificates)
./build.sh sign

# Build specific installer type
./build.sh installers msi
./build.sh installers dmg
```

### Using Python Build Script
```bash
# Build with advanced options
cd desktop
python3 build_installer.py --verbose --installer-targets msi nsis --code-signing

# Generate update manifests
python3 build_installer.py --generate-update-manifest

# Build for specific architecture
python3 build_installer.py --platform windows --installer-targets msi
```

## Platform-Specific Installers

### Windows Installers

#### MSI (Windows Installer)
- **Format**: Microsoft Installer package
- **Features**: 
  - Silent installation support
  - Add/Remove Programs integration
  - Upgrade/downgrade control
  - Registry integration
- **Configuration**: See `tauri.conf.json` → `bundle.targets[name=msi]`
- **Assets**: `banner.bmp` (493x58), `dialog.bmp` (493x312)

#### NSIS (Nullsoft Scriptable Install System)
- **Format**: Executable installer
- **Features**:
  - Custom installer UI
  - Compression options
  - Per-user/system installation
  - Uninstaller generation
- **Configuration**: See `tauri.conf.json` → `bundle.targets[name=nsis]`
- **Assets**: `header.bmp` (150x57), `sidebar.bmp` (164x314)

#### Code Signing Requirements
```bash
# Required environment variables
export WINDOWS_CERTIFICATE_PATH="path/to/certificate.p12"
export WINDOWS_CERTIFICATE_PASSWORD="certificate-password"
export TAURI_SIGNING_PRIVATE_KEY="tauri-private-key"
export TAURI_SIGNING_PRIVATE_KEY_PASSWORD="key-password"

### macOS Installer

#### DMG (Disk Image)
- **Format**: macOS disk image
- **Features**:
  - Drag-and-drop installation
  - Custom background image
  - Automatic code signing
  - Notarization support
- **Configuration**: See `tauri.conf.json` → `bundle.targets[name=dmg]`
- **Assets**: `dmg-background.png` (540x380)

#### Code Signing Requirements
```bash
# Required environment variables
export MACOS_SIGNING_IDENTITY=\"Developer ID Application: Your Name (TEAM_ID)\"
export APPLE_ID=\"your-apple-id@example.com\"
export APPLE_PASSWORD=\"app-specific-password\"
export APPLE_TEAM_ID=\"YOUR_TEAM_ID\"
```

#### Notarization Process
1. Build and sign the DMG
2. Upload to Apple for notarization
3. Staple the notarization ticket
4. Distribute the notarized DMG

### Linux Packages

#### DEB (Debian Package)
- **Format**: Debian/Ubuntu package
- **Features**:
  - APT repository integration
  - Dependency management
  - Post-install scripts
  - Desktop integration
- **Installation**: `sudo dpkg -i ttrpg-assistant.deb`

#### RPM (Red Hat Package Manager)
- **Format**: Red Hat/Fedora/SUSE package
- **Features**:
  - YUM/DNF integration
  - Dependency resolution
  - Scriptlet support
  - SELinux compatibility
- **Installation**: `sudo rpm -i ttrpg-assistant.rpm`

#### AppImage (Universal Linux)
- **Format**: Portable application
- **Features**:
  - No installation required
  - Runs on most Linux distributions
  - Self-contained executable
  - Desktop integration
- **Usage**: `chmod +x TTRPG-Assistant.AppImage && ./TTRPG-Assistant.AppImage`

#### Code Signing Requirements
```bash
# Required environment variables
export GPG_KEY_ID=\"your-gpg-key-id\"
export GPG_PASSPHRASE=\"gpg-passphrase\"
```

## Code Signing

### Overview
All installers support code signing to ensure authenticity and prevent tampering. Code signing:
- Verifies publisher identity
- Prevents modification warnings
- Enables auto-updater security
- Required for Windows SmartScreen bypass
- Required for macOS Gatekeeper approval

### Certificate Requirements

#### Windows
- **Certificate Type**: Code Signing Certificate or EV Code Signing
- **Format**: PKCS#12 (.p12 or .pfx)
- **Providers**: DigiCert, Sectigo, GlobalSign
- **Validation**: Domain or Organization validation

#### macOS
- **Certificate Type**: Developer ID Application Certificate
- **Requirements**: Apple Developer Program membership
- **Tool**: Xcode or Apple Developer Portal
- **Additional**: App-specific password for notarization

#### Linux
- **Certificate Type**: GPG key pair
- **Generation**: `gpg --generate-key`
- **Distribution**: Public key via keyserver
- **Trust**: Web of trust or direct distribution

### Setting Up Code Signing

1. **Obtain Certificates**
   - Purchase from Certificate Authority (Windows/macOS)
   - Generate GPG key (Linux)
   - Store securely with limited access

2. **Configure Environment**
   ```bash
   # Copy and configure environment file
   cp desktop/frontend/src-tauri/.env.example desktop/frontend/src-tauri/.env
   # Edit .env with your certificate paths and passwords
   ```

3. **Validate Configuration**
   ```bash
   ./build.sh validate-config
   ```

4. **Test Signing**
   ```bash
   # Build with signing enabled
   ./build.sh sign
   ```

### CI/CD Integration
For automated builds, store certificates and passwords in secure environment variables:

```yaml
# GitHub Actions example
- name: Build Signed Installers
  env:
    TAURI_SIGNING_PRIVATE_KEY: ${{ secrets.TAURI_PRIVATE_KEY }}
    TAURI_SIGNING_PRIVATE_KEY_PASSWORD: ${{ secrets.TAURI_KEY_PASSWORD }}
    WINDOWS_CERTIFICATE_PATH: ${{ secrets.WINDOWS_CERT_PATH }}
    WINDOWS_CERTIFICATE_PASSWORD: ${{ secrets.WINDOWS_CERT_PASSWORD }}
  run: ./build.sh sign
```

## Auto-Updater System

### Overview
The auto-updater provides seamless background updates with:
- Cryptographic signature verification
- Multiple release channels (stable/beta/alpha)
- Configurable update intervals
- User consent and rollback support

### Update Manifest Generation
```bash
# Generate from GitHub release
cd desktop
python3 generate-update-manifest.py --repo Raudbjorn/MDMAI --tag latest

# Generate from local files
python3 generate-update-manifest.py --local-assets ../assets --version 1.2.0

# Generate for specific release channel
python3 generate-update-manifest.py --tag v1.2.0-beta.1
```

### Release Channels

#### Stable Channel
- **Pattern**: `v*.*.*` (e.g., v1.2.0)
- **Update Check**: Daily
- **Target Users**: Production users
- **Manifest**: `latest.json`

#### Beta Channel
- **Pattern**: `v*.*.*-beta.*` (e.g., v1.2.0-beta.1)
- **Update Check**: 4 hours
- **Target Users**: Beta testers
- **Manifest**: `latest-beta.json`

#### Alpha Channel
- **Pattern**: `v*.*.*-alpha.*` (e.g., v1.2.0-alpha.1)
- **Update Check**: 1 hour
- **Target Users**: Developers
- **Manifest**: `latest-alpha.json`

### Update Deployment

1. **Build Signed Installers**
   ```bash
   ./build.sh sign
   ```

2. **Generate Update Manifests**
   ```bash
   ./build.sh update-manifests 1.2.0
   ```

3. **Upload to Release Platform**
   - Upload installer files to GitHub Releases
   - Upload update manifests to accessible URLs
   - Update CDN or download endpoints

4. **Test Update Process**
   - Install previous version
   - Trigger update check
   - Verify signature validation
   - Test rollback functionality

## Asset Management

### Required Assets

#### Windows Assets
- `banner.bmp`: MSI installer banner (493 × 58 pixels, 24-bit BMP)
- `dialog.bmp`: MSI installer dialog (493 × 312 pixels, 24-bit BMP)
- `header.bmp`: NSIS installer header (150 × 57 pixels, 24-bit BMP)
- `sidebar.bmp`: NSIS installer sidebar (164 × 314 pixels, 24-bit BMP)

#### macOS Assets
- `dmg-background.png`: DMG background (540 × 380 pixels, PNG with alpha)

#### Application Icons
- `32x32.png`: Small icon (32 × 32 pixels)
- `128x128.png`: Medium icon (128 × 128 pixels)
- `128x128@2x.png`: Retina icon (256 × 256 pixels)
- `icon.icns`: macOS icon bundle
- `icon.ico`: Windows icon bundle

### Creating Professional Assets

1. **Design Guidelines**
   - Use consistent branding and colors
   - Ensure high contrast for accessibility
   - Test on different display densities
   - Follow platform-specific conventions

2. **Technical Requirements**
   - Exact pixel dimensions required
   - Specific color depths (24-bit BMP for Windows)
   - Optimization for file size
   - Alpha channel support where needed

3. **Asset Generation**
   ```bash
   # Generate placeholder assets (development)
   ./build.sh validate-config  # Will generate placeholders

   # Replace with professional assets
   # Place files in: desktop/frontend/src-tauri/installer-assets/
   ```

## Build System Integration

### Unified Build Script
The `build.sh` script provides comprehensive installer building:

```bash
# Basic installer builds
./build.sh installers                    # All platform installers
./build.sh installers msi               # Windows MSI only
./build.sh installers dmg true          # macOS DMG with signing
./build.sh installers signed            # All signed installers

# Advanced builds
./build.sh sign                          # Signed installers + manifests
./build.sh validate-config               # Configuration validation
./build.sh update-manifests 1.2.0       # Generate update manifests
```

### Python Build System
The `build_installer.py` script provides advanced control:

```bash
cd desktop

# Basic usage
python3 build_installer.py

# Advanced options
python3 build_installer.py \
    --verbose \
    --installer-targets msi nsis dmg \
    --code-signing \
    --generate-update-manifest

# Platform-specific builds
python3 build_installer.py --platform windows --installer-targets msi
python3 build_installer.py --platform linux --installer-targets deb rpm
```

### Configuration Files

#### Main Configuration
- `desktop/frontend/src-tauri/tauri.conf.json`: Main Tauri configuration
- `desktop/frontend/src-tauri/codesign-config.json`: Code signing settings
- `desktop/frontend/src-tauri/updater-config.json`: Auto-updater configuration

#### Environment Configuration
- `desktop/frontend/src-tauri/.env.example`: Environment template
- `desktop/frontend/src-tauri/.env`: Local environment (ignored by git)

## Troubleshooting

### Common Issues

#### Build Failures
```bash
# Check dependencies
./build.sh deps

# Validate configuration
./build.sh validate-config

# Clean and rebuild
./build.sh clean all && ./build.sh installers
```

#### Code Signing Issues
```bash
# Check signing environment
./build.sh validate-config

# Verify certificates are valid and not expired
# Check environment variable values
# Ensure certificate passwords are correct
```

#### Update Manifest Problems
```bash
# Verify manifest generator dependencies
pip install requests

# Check GitHub API access
curl -H \"Authorization: token $GITHUB_TOKEN\" \
     https://api.github.com/repos/Raudbjorn/MDMAI/releases/latest

# Validate generated JSON
jq empty update-manifests/latest.json
```

### Debug Mode
Enable verbose output for debugging:

```bash
# Unified build script
./build.sh installers all true true  # Enable signing and manifests

# Python build script
python3 build_installer.py --verbose --debug

# Individual components
TAURI_DEBUG=true npm run tauri build
```

### Log Files
Check build logs for detailed error information:
- Tauri build logs: `desktop/frontend/src-tauri/target/`
- Installer build logs: `desktop/build.log`
- Update manifest logs: `desktop/updater.log`

## Security Considerations

### Code Signing Security
- Store certificates securely with limited access
- Use Hardware Security Modules (HSM) for production
- Rotate signing certificates before expiration
- Monitor certificate usage and unauthorized access

### Update Security
- Always verify cryptographic signatures
- Use HTTPS for all update endpoints
- Implement certificate pinning
- Enable rollback mechanisms for failed updates

### Distribution Security
- Use secure channels for installer distribution
- Provide checksums/hashes for manual verification
- Monitor for unauthorized redistribution
- Implement tamper detection mechanisms

## Production Deployment

### Pre-Release Checklist
- [ ] All tests passing
- [ ] Code signing certificates valid
- [ ] Professional assets in place
- [ ] Update manifests generated
- [ ] Security scan completed
- [ ] Installation tested on clean systems

### Release Process
1. **Build signed installers**: `./build.sh sign`
2. **Generate update manifests**: `./build.sh update-manifests $VERSION`
3. **Create GitHub release** with installers and manifests
4. **Update download links** on website/documentation
5. **Announce release** through appropriate channels
6. **Monitor for issues** and prepare hotfixes if needed

### Post-Release Monitoring
- Monitor crash reports and error logs
- Track update adoption rates
- Collect user feedback on installation experience
- Plan next release cycle based on feedback

## Additional Resources

### Documentation
- [Tauri Bundle Configuration](https://tauri.app/v1/api/config#bundleconfig)
- [Code Signing Best Practices](https://docs.microsoft.com/en-us/windows/win32/seccrypto/cryptography-tools)
- [Apple Notarization Guide](https://developer.apple.com/documentation/notarization)

### Tools
- [Tauri CLI](https://github.com/tauri-apps/tauri)
- [PyOxidizer](https://github.com/indygreg/PyOxidizer) (Python backend packaging)
- [GitHub Actions](https://github.com/features/actions) (CI/CD automation)

### Support
- GitHub Issues: https://github.com/Raudbjorn/MDMAI/issues
- Tauri Discord: https://discord.com/invite/tauri
- Documentation: See project README and docs/ directory