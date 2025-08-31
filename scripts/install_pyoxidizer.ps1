#Requires -Version 5.1
<#
.SYNOPSIS
    PyOxidizer Installation Script for Windows - MDMAI Project

.DESCRIPTION
    This script automates the installation of PyOxidizer on Windows systems.
    PyOxidizer is required to build standalone executables of the MDMAI MCP Server.

.PARAMETER Force
    Force reinstallation even if PyOxidizer is already installed

.EXAMPLE
    .\install_pyoxidizer.ps1
    
.EXAMPLE
    .\install_pyoxidizer.ps1 -Force
#>

param(
    [switch]$Force
)

# Enable strict mode for better error handling
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Color functions for output
function Write-Info {
    param([string]$Message)
    Write-Host "INFO: $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "SUCCESS: $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "WARNING: $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "ERROR: $Message" -ForegroundColor Red
}

# Function to test if command exists
function Test-Command {
    param([string]$Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Function to get architecture
function Get-Architecture {
    $arch = $env:PROCESSOR_ARCHITECTURE
    switch ($arch) {
        "AMD64" { return "x86_64" }
        "ARM64" { return "aarch64" }
        "x86" { return "i686" }
        default { return "unknown" }
    }
}

# Function to install Chocolatey
function Install-Chocolatey {
    Write-Info "Checking for Chocolatey..."
    
    if (Test-Command "choco") {
        Write-Success "Chocolatey is already installed"
        return
    }
    
    Write-Info "Installing Chocolatey..."
    
    # Check execution policy
    $executionPolicy = Get-ExecutionPolicy
    if ($executionPolicy -eq "Restricted") {
        Write-Warning "PowerShell execution policy is Restricted"
        Write-Info "Temporarily allowing script execution..."
        Set-ExecutionPolicy Bypass -Scope Process -Force
    }
    
    # Install Chocolatey
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    
    # Refresh environment variables
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    
    Write-Success "Chocolatey installed successfully"
}

# Function to install Rust
function Install-Rust {
    Write-Info "Checking for Rust installation..."
    
    if ((Test-Command "rustc") -and (Test-Command "cargo")) {
        $rustVersion = & rustc --version
        Write-Success "Rust is already installed: $rustVersion"
        return
    }
    
    Write-Info "Installing Rust..."
    
    # Try to install via Chocolatey first
    if (Test-Command "choco") {
        try {
            choco install rust -y
            Write-Success "Rust installed via Chocolatey"
            
            # Refresh environment variables
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
            return
        }
        catch {
            Write-Warning "Failed to install Rust via Chocolatey, trying manual installation..."
        }
    }
    
    # Manual installation via rustup-init.exe
    Write-Info "Downloading rustup-init.exe..."
    
    $tempDir = [System.IO.Path]::GetTempPath()
    $rustupPath = Join-Path $tempDir "rustup-init.exe"
    
    try {
        Invoke-WebRequest -Uri "https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe" -OutFile $rustupPath
        
        Write-Info "Running Rust installer..."
        Start-Process -FilePath $rustupPath -ArgumentList "-y" -Wait -NoNewWindow
        
        # Refresh environment variables
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        
        # Source cargo environment
        $cargoEnv = Join-Path $env:USERPROFILE ".cargo\env"
        if (Test-Path $cargoEnv) {
            & $cargoEnv
        }
        
        Write-Success "Rust installed successfully"
    }
    finally {
        if (Test-Path $rustupPath) {
            Remove-Item $rustupPath -Force
        }
    }
}

# Function to install Microsoft C++ Build Tools
function Install-BuildTools {
    Write-Info "Checking for Microsoft C++ Build Tools..."
    
    # Check for common Visual Studio installations
    $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    
    if (Test-Path $vsWhere) {
        $installations = & $vsWhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -format json | ConvertFrom-Json
        if ($installations.Count -gt 0) {
            Write-Success "Microsoft C++ Build Tools are already installed"
            return
        }
    }
    
    Write-Warning "Microsoft C++ Build Tools not detected"
    Write-Info "You need to install Microsoft C++ Build Tools manually:"
    Write-Info "1. Visit: https://visualstudio.microsoft.com/visual-cpp-build-tools/"
    Write-Info "2. Download 'Build Tools for Visual Studio'"
    Write-Info "3. Run the installer and select 'C++ build tools'"
    Write-Info "4. Make sure to include 'Windows 10/11 SDK' and 'CMake tools'"
    
    $response = Read-Host "Have you already installed Microsoft C++ Build Tools? (y/N)"
    if ($response -notmatch '^[Yy]$') {
        Write-Error "Microsoft C++ Build Tools are required for building PyOxidizer"
        exit 1
    }
}

# Function to install PyOxidizer via cargo
function Install-PyOxidizerCargo {
    Write-Info "Installing PyOxidizer via cargo..."
    
    try {
        & cargo install pyoxidizer
        Write-Success "PyOxidizer installed via cargo"
    }
    catch {
        Write-Error "Failed to install PyOxidizer via cargo: $_"
        throw
    }
}

# Function to install PyOxidizer from pre-built binary
function Install-PyOxidizerBinary {
    Write-Info "Installing PyOxidizer from pre-built binary..."
    
    $version = "0.24.0"
    $arch = Get-Architecture
    
    if ($arch -ne "x86_64") {
        Write-Warning "No pre-built binary for Windows-$arch, falling back to cargo install"
        Install-PyOxidizerCargo
        return
    }
    
    $binaryUrl = "https://github.com/indygreg/PyOxidizer/releases/download/pyoxidizer%2F$version/pyoxidizer-$version-x86_64-pc-windows-msvc.zip"
    
    Write-Info "Downloading PyOxidizer binary from $binaryUrl"
    
    $tempDir = [System.IO.Path]::GetTempPath()
    $zipPath = Join-Path $tempDir "pyoxidizer.zip"
    $extractDir = Join-Path $tempDir "pyoxidizer_extract"
    
    try {
        # Download binary
        Invoke-WebRequest -Uri $binaryUrl -OutFile $zipPath
        
        # Extract binary
        Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force
        
        # Find the executable
        $exePath = Get-ChildItem -Path $extractDir -Name "pyoxidizer.exe" -Recurse | Select-Object -First 1
        if (-not $exePath) {
            throw "Could not find pyoxidizer.exe in the downloaded archive"
        }
        
        # Install to user's local bin directory
        $installDir = Join-Path $env:USERPROFILE ".local\bin"
        New-Item -ItemType Directory -Path $installDir -Force | Out-Null
        
        $sourcePath = Join-Path $extractDir $exePath
        $destPath = Join-Path $installDir "pyoxidizer.exe"
        Copy-Item -Path $sourcePath -Destination $destPath -Force
        
        Write-Success "PyOxidizer binary installed to $installDir"
        
        # Add to PATH if not already there
        $userPath = [System.Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::User)
        if ($userPath -notlike "*$installDir*") {
            Write-Info "Adding $installDir to user PATH..."
            $newPath = "$userPath;$installDir"
            [System.Environment]::SetEnvironmentVariable("Path", $newPath, [System.EnvironmentVariableTarget]::User)
            $env:Path = "$env:Path;$installDir"
            Write-Success "Added to PATH. You may need to restart your terminal."
        }
    }
    finally {
        # Clean up
        if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
        if (Test-Path $extractDir) { Remove-Item $extractDir -Recurse -Force }
    }
}

# Function to verify PyOxidizer installation
function Test-PyOxidizerInstallation {
    Write-Info "Verifying PyOxidizer installation..."
    
    # Refresh PATH from environment
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    
    if (Test-Command "pyoxidizer") {
        $version = & pyoxidizer --version
        Write-Success "PyOxidizer is installed and available: $version"
        return $true
    }
    else {
        Write-Error "PyOxidizer is not available in PATH"
        return $false
    }
}

# Main function
function Main {
    Write-Host "===========================================" -ForegroundColor Cyan
    Write-Host "PyOxidizer Installation Script for MDMAI" -ForegroundColor Cyan
    Write-Host "===========================================" -ForegroundColor Cyan
    
    $arch = Get-Architecture
    Write-Info "Detected architecture: Windows-$arch"
    
    if ($arch -eq "unknown") {
        Write-Error "Unsupported architecture: $arch"
        exit 1
    }
    
    # Check if PyOxidizer is already installed
    if ((Test-Command "pyoxidizer") -and (-not $Force)) {
        $currentVersion = & pyoxidizer --version
        Write-Info "PyOxidizer is already installed: $currentVersion"
        $response = Read-Host "Do you want to reinstall? (y/N)"
        if ($response -notmatch '^[Yy]$') {
            Write-Info "Installation cancelled by user"
            exit 0
        }
    }
    
    try {
        # Install Chocolatey (optional, for easier dependency management)
        Install-Chocolatey
        
        # Install build dependencies
        Install-BuildTools
        
        # Install Rust
        Install-Rust
        
        # Install PyOxidizer
        Write-Info "Choose installation method:"
        Write-Host "1) Pre-built binary (recommended, faster)"
        Write-Host "2) Cargo install (requires compilation)"
        
        $choice = Read-Host "Enter choice (1-2) [1]"
        
        switch ($choice) {
            "2" { Install-PyOxidizerCargo }
            default { Install-PyOxidizerBinary }
        }
        
        # Verify installation
        if (Test-PyOxidizerInstallation) {
            Write-Success "PyOxidizer installation completed successfully!"
            Write-Host ""
            Write-Info "Next steps:"
            Write-Host "1. Navigate to your MDMAI project directory"
            Write-Host "2. Run: python scripts\build_pyoxidizer.py --platform windows"
            Write-Host "3. Or run: python scripts\build_pyoxidizer.py --all (for all platforms)"
        }
        else {
            Write-Error "Installation verification failed"
            exit 1
        }
    }
    catch {
        Write-Error "Installation failed: $_"
        exit 1
    }
}

# Run main function
Main