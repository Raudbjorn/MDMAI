# TTRPG Assistant Desktop Application - Deployment Guide

## Table of Contents
1. [Overview](#overview)
2. [Building from Source](#building-from-source)
3. [Platform-Specific Builds](#platform-specific-builds)
4. [Code Signing](#code-signing)
5. [Creating Installers](#creating-installers)
6. [Distribution](#distribution)
7. [Auto-Updates](#auto-updates)
8. [CI/CD Pipeline](#cicd-pipeline)
9. [Enterprise Deployment](#enterprise-deployment)
10. [Troubleshooting](#troubleshooting)

## Overview

This guide covers the complete deployment process for the TTRPG Assistant Desktop Application, from building the source code to distributing signed installers.

### Deployment Architecture

```
Source Code → Build → Bundle Python → Package → Sign → Distribute
     ↓           ↓          ↓            ↓        ↓         ↓
   GitHub     Tauri    PyOxidizer    Installers  Certs   Releases
```

### Prerequisites

**Development Tools:**
- Node.js 18+ and npm/pnpm
- Rust 1.70+ and Cargo
- Python 3.11+
- Git

**Platform-Specific:**
- **Windows**: Visual Studio 2022 Build Tools, WiX Toolset 3.11
- **macOS**: Xcode Command Line Tools
- **Linux**: build-essential, libgtk-3-dev, libwebkit2gtk-4.0-dev

## Building from Source

### 1. Clone and Setup

```bash
# Clone repository
git clone https://github.com/Raudbjorn/MDMAI.git
cd MDMAI

# Install dependencies
./build.sh setup

# Or manually:
# Python dependencies
cd desktop/backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend dependencies
cd ../frontend
npm install

# Rust dependencies
cargo build --release
```

### 2. Development Build

```bash
# Quick development build
./build.sh desktop

# Or manually:
cd desktop/frontend
npm run tauri:build:debug
```

### 3. Release Build

```bash
# Production build with optimizations
./build.sh desktop-release

# Or manually:
cd desktop/frontend
npm run tauri:build
```

## Platform-Specific Builds

### Windows Build

```powershell
# Prerequisites
winget install Microsoft.VisualStudio.2022.BuildTools
winget install WiX.Toolset

# Build
cd desktop/frontend
npm run tauri:build -- --target x86_64-pc-windows-msvc

# Output: src-tauri/target/release/bundle/msi/TTRPG Assistant_*.msi
```

**Windows-Specific Configuration:**
```json
# src-tauri/tauri.conf.json
{
  "tauri": {
    "bundle": {
      "windows": {
        "certificateThumbprint": "YOUR_CERT_THUMBPRINT",
        "digestAlgorithm": "sha256",
        "timestampUrl": "https://timestamp.digicert.com",
        "wix": {
          "language": "en-US",
          "template": "wix/template.wxs"
        }
      }
    }
  }
}
```

### macOS Build

```bash
# Prerequisites
xcode-select --install

# Build for Intel
npm run tauri:build -- --target x86_64-apple-darwin

# Build for Apple Silicon
npm run tauri:build -- --target aarch64-apple-darwin

# Universal binary
npm run tauri:build -- --target universal-apple-darwin

# Output: src-tauri/target/release/bundle/dmg/TTRPG Assistant_*.dmg
```

**macOS-Specific Configuration:**
```json
{
  "tauri": {
    "bundle": {
      "macOS": {
        "frameworks": [],
        "minimumSystemVersion": "10.15",
        "exceptionDomain": "",
        "signingIdentity": "Developer ID Application: Your Name (TEAM_ID)",
        "providerShortName": "TEAM_ID",
        "entitlements": "entitlements.plist"
      }
    }
  }
}
```

### Linux Build

```bash
# Debian/Ubuntu prerequisites
sudo apt update
sudo apt install libgtk-3-dev libwebkit2gtk-4.0-dev \
    libssl-dev libgtk-3-0 libwebkit2gtk-4.0-37 \
    libayatana-appindicator3-dev librsvg2-dev

# Build AppImage
npm run tauri:build -- --target x86_64-unknown-linux-gnu
# Output: src-tauri/target/release/bundle/appimage/ttrpg-assistant_*.AppImage

# Build .deb
npm run tauri:build -- --bundles deb
# Output: src-tauri/target/release/bundle/deb/ttrpg-assistant_*.deb

# Build .rpm
npm run tauri:build -- --bundles rpm
# Output: src-tauri/target/release/bundle/rpm/ttrpg-assistant-*.rpm
```

## Code Signing

### Windows Code Signing

1. **Obtain Certificate:**
   ```powershell
   # Self-signed for testing
   New-SelfSignedCertificate -Type CodeSigningCert -Subject "CN=Your Company" -KeyExportPolicy Exportable -KeySpec Signature -KeyLength 2048 -KeyAlgorithm RSA -HashAlgorithm SHA256 -CertStoreLocation "Cert:\CurrentUser\My"
   
   # Export certificate
   $cert = Get-ChildItem -Path Cert:\CurrentUser\My -CodeSigningCert
   Export-PfxCertificate -Cert $cert -FilePath mycert.pfx -Password (ConvertTo-SecureString -String "password" -Force -AsPlainText)
   ```

2. **Sign Application:**
   ```powershell
   # Using SignTool
   signtool sign /f mycert.pfx /p password /fd sha256 /tr https://timestamp.digicert.com /td sha256 "TTRPG Assistant.exe"
   ```

3. **Verify Signature:**
   ```powershell
   signtool verify /pa "TTRPG Assistant.exe"
   ```

### macOS Code Signing

1. **Developer ID Certificate:**
   - Enroll in Apple Developer Program
   - Download from developer.apple.com/account

2. **Sign Application:**
   ```bash
   # Sign app
   codesign --deep --force --verbose --sign "Developer ID Application: Your Name (TEAM_ID)" "TTRPG Assistant.app"
   
   # Sign DMG
   codesign --sign "Developer ID Application: Your Name (TEAM_ID)" "TTRPG Assistant.dmg"
   ```

3. **Notarization:**
   ```bash
   # Submit for notarization
   xcrun altool --notarize-app --primary-bundle-id "com.yourcompany.ttrpg-assistant" --username "apple-id@example.com" --password "app-specific-password" --file "TTRPG Assistant.dmg"
   
   # Check status
   xcrun altool --notarization-info REQUEST_ID --username "apple-id@example.com"
   
   # Staple ticket
   xcrun stapler staple "TTRPG Assistant.dmg"
   ```

### Linux Code Signing

```bash
# GPG signing for packages
gpg --gen-key
gpg --armor --export your-email@example.com > public.key

# Sign .deb package
dpkg-sig --sign builder package.deb

# Sign .rpm package
rpm --addsign package.rpm

# Sign AppImage (optional)
appimagetool --sign TTRPG-Assistant.AppImage
```

## Creating Installers

### Windows Installer (WiX)

```xml
<!-- wix/template.wxs -->
<?xml version="1.0" encoding="UTF-8"?>
<Wix xmlns="http://schemas.microsoft.com/wix/2006/wi">
    <Product Id="*" 
             Name="TTRPG Assistant" 
             Language="1033" 
             Version="{{version}}" 
             Manufacturer="Your Company"
             UpgradeCode="YOUR-UPGRADE-CODE">
        
        <Package InstallerVersion="300" Compressed="yes" InstallScope="perMachine" />
        
        <MajorUpgrade DowngradeErrorMessage="A newer version is already installed." />
        
        <Feature Id="ProductFeature" Title="TTRPG Assistant" Level="1">
            <ComponentGroupRef Id="ProductComponents" />
        </Feature>
        
        <!-- Desktop shortcut -->
        <DirectoryRef Id="DesktopFolder">
            <Component Id="DesktopShortcut" Guid="YOUR-GUID">
                <Shortcut Id="ApplicationDesktopShortcut"
                         Name="TTRPG Assistant"
                         Description="TTRPG Assistant Desktop Application"
                         Target="[INSTALLDIR]TTRPG Assistant.exe"
                         WorkingDirectory="INSTALLDIR"/>
            </Component>
        </DirectoryRef>
    </Product>
</Wix>
```

Build installer:
```powershell
# Automated through Tauri
npm run tauri:build

# Or manually with WiX
candle.exe template.wxs -o template.wixobj
light.exe -ext WixUIExtension template.wixobj -o "TTRPG Assistant.msi"
```

### macOS DMG

```bash
# Create DMG with custom background
create-dmg \
  --volname "TTRPG Assistant" \
  --volicon "icon.icns" \
  --background "installer-background.png" \
  --window-pos 200 120 \
  --window-size 600 400 \
  --icon-size 100 \
  --icon "TTRPG Assistant.app" 175 120 \
  --hide-extension "TTRPG Assistant.app" \
  --app-drop-link 425 120 \
  "TTRPG Assistant.dmg" \
  "dist/"
```

### Linux Packages

**AppImage:**
```bash
# Automated through Tauri
npm run tauri:build -- --bundles appimage

# Manual with appimagetool
ARCH=x86_64 appimagetool TTRPG-Assistant.AppDir
```

**Debian Package:**
```bash
# Create package structure
mkdir -p debian/DEBIAN
mkdir -p debian/usr/bin
mkdir -p debian/usr/share/applications

# Create control file
cat > debian/DEBIAN/control << EOF
Package: ttrpg-assistant
Version: 1.0.0
Architecture: amd64
Maintainer: Your Name <email@example.com>
Description: TTRPG Assistant Desktop Application
Depends: libgtk-3-0, libwebkit2gtk-4.0-37
EOF

# Build package
dpkg-deb --build debian ttrpg-assistant.deb
```

## Distribution

### GitHub Releases

1. **Automated Release Workflow:**
```yaml
# .github/workflows/release.yml
name: Release
on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
    
    runs-on: ${{ matrix.os }}
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node
        uses: actions/setup-node@v3
        with:
          node-version: 18
      
      - name: Setup Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      
      - name: Build
        run: |
          cd desktop/frontend
          npm install
          npm run tauri:build
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: ${{ matrix.os }}-release
          path: |
            desktop/frontend/src-tauri/target/release/bundle/
```

2. **Create Release:**
```bash
# Tag version
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# GitHub CLI
gh release create v1.0.0 \
  --title "TTRPG Assistant v1.0.0" \
  --notes "Release notes here" \
  ./dist/TTRPG-Assistant-*.msi \
  ./dist/TTRPG-Assistant-*.dmg \
  ./dist/TTRPG-Assistant-*.AppImage
```

### Web Distribution

**Download Page:**
```html
<!-- downloads.html -->
<div class="downloads">
  <div class="platform" data-os="windows">
    <h3>Windows</h3>
    <a href="https://github.com/Raudbjorn/MDMAI/releases/latest/download/TTRPG-Assistant-Setup.exe" 
       class="button primary">Download for Windows</a>
    <p>Windows 10 or later • 64-bit</p>
  </div>
  
  <div class="platform" data-os="macos">
    <h3>macOS</h3>
    <a href="https://github.com/Raudbjorn/MDMAI/releases/latest/download/TTRPG-Assistant.dmg"
       class="button primary">Download for macOS</a>
    <p>macOS 10.15 or later</p>
  </div>
  
  <div class="platform" data-os="linux">
    <h3>Linux</h3>
    <a href="https://github.com/Raudbjorn/MDMAI/releases/latest/download/TTRPG-Assistant.AppImage"
       class="button primary">Download AppImage</a>
    <p>Most Linux distributions</p>
  </div>
</div>
```

### Package Managers

**Windows (Chocolatey):**
```powershell
# Create package
choco pack ttrpg-assistant.nuspec

# Submit to community repository
choco push ttrpg-assistant.1.0.0.nupkg --source https://push.chocolatey.org/
```

**macOS (Homebrew):**
```ruby
# Formula: ttrpg-assistant.rb
class TtrpgAssistant < Formula
  desc "TTRPG Assistant Desktop Application"
  homepage "https://github.com/Raudbjorn/MDMAI"
  url "https://github.com/Raudbjorn/MDMAI/releases/download/v1.0.0/TTRPG-Assistant-1.0.0.tar.gz"
  sha256 "SHA256_HASH_HERE"
  license "MIT"

  def install
    bin.install "ttrpg-assistant"
  end
end
```

**Linux (Snap):**
```yaml
# snapcraft.yaml
name: ttrpg-assistant
version: '1.0.0'
summary: TTRPG Assistant Desktop Application
description: |
  AI-powered assistant for tabletop role-playing games

grade: stable
confinement: strict

parts:
  ttrpg-assistant:
    plugin: dump
    source: .
    
apps:
  ttrpg-assistant:
    command: ttrpg-assistant
    plugs:
      - desktop
      - desktop-legacy
      - home
      - network
```

## Auto-Updates

### Tauri Updater Configuration

```json
// tauri.conf.json
{
  "tauri": {
    "updater": {
      "active": true,
      "endpoints": [
        "https://releases.ttrpg-assistant.com/update/{{target}}/{{current_version}}"
      ],
      "dialog": true,
      "pubkey": "YOUR_PUBLIC_KEY"
    }
  }
}
```

### Update Server

```javascript
// update-server.js
const express = require('express');
const app = express();

app.get('/update/:target/:version', (req, res) => {
  const { target, version } = req.params;
  
  // Check for updates
  const latestVersion = getLatestVersion(target);
  
  if (compareVersions(latestVersion, version) > 0) {
    res.json({
      version: latestVersion,
      notes: getReleaseNotes(latestVersion),
      pub_date: new Date().toISOString(),
      platforms: {
        [target]: {
          signature: getSignature(target, latestVersion),
          url: getDownloadUrl(target, latestVersion)
        }
      }
    });
  } else {
    res.status(204).send();
  }
});

app.listen(3000);
```

### Update Manifest

```json
// update-manifest.json
{
  "version": "1.0.1",
  "notes": "Bug fixes and performance improvements",
  "pub_date": "2024-01-15T10:00:00Z",
  "platforms": {
    "darwin-x86_64": {
      "signature": "SIGNATURE_HERE",
      "url": "https://github.com/Raudbjorn/MDMAI/releases/download/v1.0.1/TTRPG-Assistant.app.tar.gz"
    },
    "windows-x86_64": {
      "signature": "SIGNATURE_HERE",
      "url": "https://github.com/Raudbjorn/MDMAI/releases/download/v1.0.1/TTRPG-Assistant.msi.zip"
    },
    "linux-x86_64": {
      "signature": "SIGNATURE_HERE",
      "url": "https://github.com/Raudbjorn/MDMAI/releases/download/v1.0.1/TTRPG-Assistant.AppImage.tar.gz"
    }
  }
}
```

## CI/CD Pipeline

### GitHub Actions Complete Pipeline

```yaml
# .github/workflows/build-and-release.yml
name: Build and Release

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:

jobs:
  build:
    strategy:
      fail-fast: false
      matrix:
        platform:
          - os: ubuntu-latest
            target: x86_64-unknown-linux-gnu
            bundles: appimage,deb
          - os: windows-latest
            target: x86_64-pc-windows-msvc
            bundles: msi,nsis
          - os: macos-latest
            target: x86_64-apple-darwin
            bundles: dmg
          - os: macos-latest
            target: aarch64-apple-darwin
            bundles: dmg
    
    runs-on: ${{ matrix.platform.os }}
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node
        uses: actions/setup-node@v3
        with:
          node-version: 18
      
      - name: Setup Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          target: ${{ matrix.platform.target }}
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      # Linux dependencies
      - name: Install Linux dependencies
        if: matrix.platform.os == 'ubuntu-latest'
        run: |
          sudo apt-get update
          sudo apt-get install -y libgtk-3-dev libwebkit2gtk-4.0-dev \
            libssl-dev libayatana-appindicator3-dev librsvg2-dev
      
      # Build Python bundle
      - name: Build Python bundle
        run: |
          cd desktop/backend
          python -m venv venv
          source venv/bin/activate || venv\\Scripts\\activate
          pip install pyoxidizer
          pyoxidizer build --release
      
      # Build Tauri app
      - name: Build Tauri app
        run: |
          cd desktop/frontend
          npm ci
          npm run tauri:build -- --target ${{ matrix.platform.target }} --bundles ${{ matrix.platform.bundles }}
      
      # Code signing (Windows)
      - name: Sign Windows binaries
        if: matrix.platform.os == 'windows-latest'
        run: |
          # Import certificate
          $pfx_cert_byte = [System.Convert]::FromBase64String("${{ secrets.WINDOWS_CERTIFICATE }}")
          $cert_path = "cert.pfx"
          [IO.File]::WriteAllBytes($cert_path, $pfx_cert_byte)
          
          # Sign executable
          & signtool sign /f $cert_path /p "${{ secrets.WINDOWS_CERTIFICATE_PASSWORD }}" /fd sha256 /tr https://timestamp.digicert.com /td sha256 "desktop/frontend/src-tauri/target/release/TTRPG Assistant.exe"
      
      # Code signing (macOS)
      - name: Sign macOS app
        if: startsWith(matrix.platform.os, 'macos')
        env:
          APPLE_CERTIFICATE: ${{ secrets.APPLE_CERTIFICATE }}
          APPLE_CERTIFICATE_PASSWORD: ${{ secrets.APPLE_CERTIFICATE_PASSWORD }}
          APPLE_SIGNING_IDENTITY: ${{ secrets.APPLE_SIGNING_IDENTITY }}
          APPLE_ID: ${{ secrets.APPLE_ID }}
          APPLE_PASSWORD: ${{ secrets.APPLE_PASSWORD }}
        run: |
          # Import certificate
          echo $APPLE_CERTIFICATE | base64 --decode > certificate.p12
          security create-keychain -p actions temp.keychain
          security import certificate.p12 -k temp.keychain -P $APPLE_CERTIFICATE_PASSWORD -T /usr/bin/codesign
          security set-key-partition-list -S apple-tool:,apple: -s -k actions temp.keychain
          
          # Sign app
          codesign --deep --force --verbose --sign "$APPLE_SIGNING_IDENTITY" "desktop/frontend/src-tauri/target/release/bundle/macos/TTRPG Assistant.app"
          
          # Notarize
          xcrun altool --notarize-app --primary-bundle-id "com.ttrpg.assistant" --username "$APPLE_ID" --password "$APPLE_PASSWORD" --file "desktop/frontend/src-tauri/target/release/bundle/dmg/TTRPG Assistant.dmg"
      
      # Upload artifacts
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: ${{ matrix.platform.target }}
          path: desktop/frontend/src-tauri/target/release/bundle/
  
  release:
    needs: build
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Download artifacts
        uses: actions/download-artifact@v3
      
      - name: Create release
        uses: softprops/action-gh-release@v1
        with:
          files: |
            **/TTRPG-Assistant*.msi
            **/TTRPG-Assistant*.exe
            **/TTRPG-Assistant*.dmg  
            **/TTRPG-Assistant*.AppImage
            **/ttrpg-assistant*.deb
            **/ttrpg-assistant*.rpm
          draft: false
          prerelease: false
          generate_release_notes: true
```

## Enterprise Deployment

### Group Policy (Windows)

```xml
<!-- TTRPG-Assistant.admx -->
<?xml version="1.0" encoding="utf-8"?>
<policyDefinitions xmlns:xsd="http://www.w3.org/2001/XMLSchema" 
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
                   xmlns="http://schemas.microsoft.com/GroupPolicy/2006/07/PolicyDefinitions">
  <policyNamespaces>
    <target prefix="ttrpg" namespace="TTRPG.Assistant.Policies" />
  </policyNamespaces>
  
  <policies>
    <policy name="AutoUpdate" class="Machine" displayName="Auto Update" 
            explainText="Enable or disable automatic updates">
      <parentCategory ref="TTRPG_Assistant" />
      <supportedOn ref="windows:SUPPORTED_Windows10" />
      <elements>
        <boolean id="AutoUpdateEnabled" valueName="AutoUpdate">
          <trueValue><decimal value="1" /></trueValue>
          <falseValue><decimal value="0" /></falseValue>
        </boolean>
      </elements>
    </policy>
  </policies>
</policyDefinitions>
```

### MDM Configuration (macOS)

```xml
<!-- com.ttrpg.assistant.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>PayloadContent</key>
    <array>
        <dict>
            <key>PayloadType</key>
            <string>com.ttrpg.assistant</string>
            <key>PayloadVersion</key>
            <integer>1</integer>
            <key>PayloadIdentifier</key>
            <string>com.ttrpg.assistant.config</string>
            <key>PayloadUUID</key>
            <string>YOUR-UUID</string>
            <key>PayloadDisplayName</key>
            <string>TTRPG Assistant Configuration</string>
            <key>AutoUpdate</key>
            <true/>
            <key>DataDirectory</key>
            <string>/Users/Shared/TTRPG-Assistant</string>
        </dict>
    </array>
</dict>
</plist>
```

### Silent Installation

**Windows:**
```batch
:: Silent install
msiexec /i "TTRPG Assistant.msi" /quiet /norestart

:: With logging
msiexec /i "TTRPG Assistant.msi" /quiet /norestart /log install.log

:: Custom directory
msiexec /i "TTRPG Assistant.msi" INSTALLDIR="C:\Apps\TTRPG" /quiet
```

**macOS:**
```bash
# Silent install from DMG
hdiutil attach TTRPG-Assistant.dmg -nobrowse -quiet
cp -R "/Volumes/TTRPG Assistant/TTRPG Assistant.app" /Applications/
hdiutil detach "/Volumes/TTRPG Assistant" -quiet

# Using installer package
installer -pkg TTRPG-Assistant.pkg -target /
```

**Linux:**
```bash
# Debian/Ubuntu
sudo dpkg -i ttrpg-assistant.deb

# RedHat/Fedora
sudo rpm -i ttrpg-assistant.rpm

# AppImage
chmod +x TTRPG-Assistant.AppImage
./TTRPG-Assistant.AppImage --appimage-extract
mv squashfs-root /opt/ttrpg-assistant
ln -s /opt/ttrpg-assistant/AppRun /usr/local/bin/ttrpg-assistant
```

## Troubleshooting

### Build Issues

**Rust compilation errors:**
```bash
# Clear cache
cargo clean

# Update dependencies
cargo update

# Rebuild
cargo build --release
```

**Node/npm issues:**
```bash
# Clear npm cache
npm cache clean --force

# Remove node_modules
rm -rf node_modules package-lock.json

# Reinstall
npm install
```

**Python bundling issues:**
```bash
# Update PyOxidizer
pip install --upgrade pyoxidizer

# Clear PyOxidizer cache
rm -rf build/pyoxidizer

# Rebuild
pyoxidizer build --release
```

### Signing Issues

**Windows:**
- Verify certificate is valid: `certutil -dump cert.pfx`
- Check timestamp server is accessible
- Ensure certificate has code signing capability

**macOS:**
- Verify Developer ID: `security find-identity -v -p codesigning`
- Check notarization status: `xcrun altool --notarization-history 0 -u apple-id`
- Ensure Xcode is up to date

### Distribution Issues

**Upload failures:**
```bash
# Check file size limits
ls -lh dist/

# Split large files
split -b 2GB large-file.dmg large-file.dmg.part

# Use GitHub LFS for large files
git lfs track "*.dmg"
git lfs track "*.msi"
```

**Update server issues:**
- Verify update manifest JSON is valid
- Check signatures match
- Ensure URLs are accessible
- Test with curl: `curl -I https://your-update-server/update-manifest.json`

### Performance Optimization

**Build size reduction:**
```toml
# Cargo.toml
[profile.release]
opt-level = "z"     # Optimize for size
lto = true          # Link-time optimization
codegen-units = 1   # Single codegen unit
strip = true        # Strip symbols
```

**Startup time optimization:**
```javascript
// Lazy load heavy modules
const loadHeavyModule = () => import('./heavy-module');

// Preload critical resources
window.addEventListener('DOMContentLoaded', () => {
  // Preload essential data
  preloadCriticalResources();
});
```

## Security Considerations

### Secure Distribution

1. **HTTPS Only**: Always serve downloads over HTTPS
2. **Checksum Verification**: Provide SHA256 checksums
3. **GPG Signatures**: Sign releases with GPG
4. **Certificate Pinning**: Pin update server certificates

### Runtime Security

```rust
// src-tauri/src/main.rs
fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            // Whitelist commands
        ])
        .setup(|app| {
            // Content Security Policy
            app.get_window("main").unwrap().eval(&format!(
                "window.__TAURI_METADATA__.__CSP__ = \"{}\"",
                "default-src 'self'; script-src 'self'"
            ));
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

## Monitoring and Analytics

### Crash Reporting

```javascript
// Sentry integration
import * as Sentry from "@sentry/electron";

Sentry.init({
  dsn: "YOUR_SENTRY_DSN",
  environment: process.env.NODE_ENV,
  beforeSend(event) {
    // Filter sensitive data
    return event;
  }
});
```

### Usage Analytics (Optional)

```javascript
// Privacy-respecting analytics
const analytics = {
  track(event, properties) {
    if (userConsent) {
      // Send to analytics service
      fetch('/analytics', {
        method: 'POST',
        body: JSON.stringify({ event, properties })
      });
    }
  }
};
```

## Maintenance

### Version Management

```bash
# Bump version
npm version patch  # 1.0.0 -> 1.0.1
npm version minor  # 1.0.0 -> 1.1.0
npm version major  # 1.0.0 -> 2.0.0

# Update all version references
./scripts/bump-version.sh 1.0.1
```

### Deprecation Strategy

```javascript
// Mark deprecated features
console.warn('[DEPRECATED] This feature will be removed in v2.0.0');

// Provide migration path
if (useOldFeature) {
  console.log('Please migrate to newFeature()');
  // Temporary compatibility layer
  return newFeature();
}
```

---

This deployment guide provides comprehensive instructions for building, signing, and distributing the TTRPG Assistant Desktop Application across all major platforms. Follow platform-specific sections for your target environment.