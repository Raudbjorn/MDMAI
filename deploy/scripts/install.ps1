#############################################################################
# TTRPG Assistant MCP Server - Installation Script for Windows
#############################################################################

param(
    [string]$InstallDir = "C:\Program Files\TTRPG-Assistant",
    [string]$DataDir = "C:\ProgramData\TTRPG-Assistant",
    [string]$ConfigDir = "C:\ProgramData\TTRPG-Assistant\Config",
    [string]$LogDir = "C:\ProgramData\TTRPG-Assistant\Logs",
    [string]$PythonVersion = "3.9",
    [string]$InstallMode = "standalone",  # standalone, service, docker
    [string]$GPUSupport = "none",  # none, cuda
    [switch]$Help
)

# Requires administrator privileges
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "This script requires Administrator privileges. Restarting as Administrator..." -ForegroundColor Red
    Start-Process PowerShell -Verb RunAs "-File `"$PSCommandPath`" $PSBoundParameters"
    exit
}

# Helper functions
function Write-Header {
    param([string]$Message)
    Write-Host "============================================" -ForegroundColor Blue
    Write-Host $Message -ForegroundColor Blue
    Write-Host "============================================" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

function Show-Help {
    Write-Host "TTRPG Assistant MCP Server - Windows Installation Script"
    Write-Host ""
    Write-Host "Usage: .\install.ps1 [options]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -InstallDir <path>    Installation directory (default: C:\Program Files\TTRPG-Assistant)"
    Write-Host "  -DataDir <path>       Data directory (default: C:\ProgramData\TTRPG-Assistant)"
    Write-Host "  -ConfigDir <path>     Configuration directory (default: C:\ProgramData\TTRPG-Assistant\Config)"
    Write-Host "  -LogDir <path>        Log directory (default: C:\ProgramData\TTRPG-Assistant\Logs)"
    Write-Host "  -PythonVersion <ver>  Python version to install (default: 3.9)"
    Write-Host "  -InstallMode <mode>   Installation mode: standalone, service, docker (default: standalone)"
    Write-Host "  -GPUSupport <type>    GPU support: none, cuda (default: none)"
    Write-Host "  -Help                 Show this help message"
    exit
}

if ($Help) {
    Show-Help
}

Write-Header "TTRPG Assistant MCP Server Installation for Windows"

# Check for existing Python installation
function Check-Python {
    Write-Header "Checking Python Installation"
    
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        $version = python --version 2>&1
        if ($version -match "Python (\d+\.\d+)") {
            $installedVersion = $matches[1]
            if ([version]$installedVersion -ge [version]$PythonVersion) {
                Write-Success "Python $installedVersion found"
                return $true
            } else {
                Write-Warning "Python $installedVersion found, but $PythonVersion or higher is required"
            }
        }
    }
    
    Write-Warning "Python not found or version too old. Installing Python..."
    return $false
}

# Install Python
function Install-Python {
    Write-Header "Installing Python $PythonVersion"
    
    $pythonUrl = "https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe"
    $installerPath = "$env:TEMP\python-installer.exe"
    
    Write-Host "Downloading Python installer..."
    Invoke-WebRequest -Uri $pythonUrl -OutFile $installerPath
    
    Write-Host "Installing Python..."
    Start-Process -FilePath $installerPath -ArgumentList "/quiet", "InstallAllUsers=1", "PrependPath=1" -Wait
    
    Remove-Item $installerPath
    
    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    
    Write-Success "Python installed successfully"
}

# Install Git if not present
function Check-Git {
    Write-Header "Checking Git Installation"
    
    $gitCmd = Get-Command git -ErrorAction SilentlyContinue
    if ($gitCmd) {
        Write-Success "Git found"
        return $true
    }
    
    Write-Warning "Git not found. Installing Git..."
    $gitUrl = "https://github.com/git-for-windows/git/releases/download/v2.42.0.windows.2/Git-2.42.0.2-64-bit.exe"
    $installerPath = "$env:TEMP\git-installer.exe"
    
    Invoke-WebRequest -Uri $gitUrl -OutFile $installerPath
    Start-Process -FilePath $installerPath -ArgumentList "/VERYSILENT" -Wait
    Remove-Item $installerPath
    
    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    
    Write-Success "Git installed successfully"
    return $true
}

# Create directory structure
function Create-Directories {
    Write-Header "Creating Directory Structure"
    
    $directories = @(
        $InstallDir,
        $DataDir,
        "$DataDir\chromadb",
        "$DataDir\cache",
        "$DataDir\backup",
        "$DataDir\export",
        $ConfigDir,
        $LogDir
    )
    
    foreach ($dir in $directories) {
        if (!(Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Success "Created directory: $dir"
        } else {
            Write-Warning "Directory already exists: $dir"
        }
    }
}

# Clone or copy repository
function Setup-Repository {
    Write-Header "Setting Up Repository"
    
    # Get the script's parent directory (deploy/scripts)
    $scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
    $deployPath = Split-Path -Parent $scriptPath
    $projectRoot = Split-Path -Parent $deployPath
    
    if (Test-Path "$InstallDir\.git") {
        Write-Warning "Repository already exists, pulling latest changes"
        Set-Location $InstallDir
        git pull origin main
    } else {
        Write-Host "Copying files from $projectRoot to $InstallDir"
        
        # Copy all files and folders
        Get-ChildItem -Path $projectRoot -Recurse | ForEach-Object {
            $targetPath = $_.FullName.Replace($projectRoot, $InstallDir)
            
            if ($_.PSIsContainer) {
                if (!(Test-Path $targetPath)) {
                    New-Item -ItemType Directory -Path $targetPath -Force | Out-Null
                }
            } else {
                $targetDir = Split-Path -Parent $targetPath
                if (!(Test-Path $targetDir)) {
                    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
                }
                Copy-Item -Path $_.FullName -Destination $targetPath -Force
            }
        }
        
        Write-Success "Repository files copied"
    }
}

# Setup Python virtual environment
function Setup-VirtualEnvironment {
    Write-Header "Setting Up Python Virtual Environment"
    
    Set-Location $InstallDir
    
    # Create virtual environment
    python -m venv venv
    
    # Activate and upgrade pip
    & "$InstallDir\venv\Scripts\Activate.ps1"
    python -m pip install --upgrade pip setuptools wheel
    
    Write-Success "Virtual environment created"
}

# Install Python dependencies
function Install-Dependencies {
    Write-Header "Installing Python Dependencies"
    
    Set-Location $InstallDir
    & "$InstallDir\venv\Scripts\Activate.ps1"
    
    # Install based on GPU support
    switch ($GPUSupport) {
        "cuda" {
            Write-Warning "Installing with CUDA support (this may take a while)"
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        }
        default {
            Write-Warning "Installing CPU-only version (no GPU support)"
            pip install torch torchvision torchaudio
        }
    }
    
    # Install main package
    pip install -e .
    
    # Download spaCy model
    python -m spacy download en_core_web_sm
    
    Write-Success "Python dependencies installed"
}

# Configure application
function Configure-Application {
    Write-Header "Configuring Application"
    
    # Copy configuration templates
    Copy-Item "$InstallDir\deploy\config\.env.template" "$ConfigDir\.env"
    Copy-Item "$InstallDir\deploy\config\config.yaml.template" "$ConfigDir\config.yaml"
    
    # Update configuration paths (Windows style)
    $envContent = Get-Content "$ConfigDir\.env"
    $envContent = $envContent -replace 'CHROMA_DB_PATH=.*', "CHROMA_DB_PATH=$($DataDir.Replace('\', '\\'))\\chromadb"
    $envContent = $envContent -replace 'CACHE_DIR=.*', "CACHE_DIR=$($DataDir.Replace('\', '\\'))\\cache"
    $envContent = $envContent -replace 'LOG_FILE=.*', "LOG_FILE=$($LogDir.Replace('\', '\\'))\\ttrpg-assistant.log"
    $envContent = $envContent -replace 'SECURITY_LOG_FILE=.*', "SECURITY_LOG_FILE=$($LogDir.Replace('\', '\\'))\\security.log"
    Set-Content "$ConfigDir\.env" $envContent
    
    Write-Success "Application configured"
}

# Install Windows Service
function Install-WindowsService {
    if ($InstallMode -ne "service") {
        return
    }
    
    Write-Header "Installing Windows Service"
    
    # Create a service wrapper script
    $serviceScript = @"
import sys
import os
import time
import win32serviceutil
import win32service
import win32event
import servicemanager
import socket

sys.path.insert(0, r'$InstallDir')
os.environ['CONFIG_DIR'] = r'$ConfigDir'

class TTRPGAssistantService(win32serviceutil.ServiceFramework):
    _svc_name_ = 'TTRPGAssistant'
    _svc_display_name_ = 'TTRPG Assistant MCP Server'
    _svc_description_ = 'MCP server for TTRPG assistance'
    
    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        socket.setdefaulttimeout(60)
        
    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)
        
    def SvcDoRun(self):
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, '')
        )
        self.main()
        
    def main(self):
        # Import and run the main application
        from src.main import main
        main()

if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(TTRPGAssistantService)
"@
    
    Set-Content -Path "$InstallDir\service_wrapper.py" -Value $serviceScript
    
    # Install pywin32
    & "$InstallDir\venv\Scripts\Activate.ps1"
    pip install pywin32
    
    # Install the service
    python "$InstallDir\service_wrapper.py" install
    
    Write-Success "Windows service installed"
}

# Setup Docker (if Docker Desktop is installed)
function Setup-Docker {
    if ($InstallMode -ne "docker") {
        return
    }
    
    Write-Header "Setting Up Docker"
    
    # Check if Docker is installed
    $dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
    if (!$dockerCmd) {
        Write-Error "Docker Desktop is not installed. Please install Docker Desktop first."
        Write-Host "Download from: https://www.docker.com/products/docker-desktop"
        return
    }
    
    # Copy Docker Compose template
    Copy-Item "$InstallDir\deploy\config\docker-compose.yaml.template" "$InstallDir\docker-compose.yaml"
    
    # Update paths in docker-compose.yaml
    $composeContent = Get-Content "$InstallDir\docker-compose.yaml"
    $composeContent = $composeContent -replace '{{DATA_DIR}}', $DataDir.Replace('\', '/')
    $composeContent = $composeContent -replace '{{CONFIG_DIR}}', $ConfigDir.Replace('\', '/')
    $composeContent = $composeContent -replace '{{LOG_DIR}}', $LogDir.Replace('\', '/')
    Set-Content "$InstallDir\docker-compose.yaml" $composeContent
    
    # Build Docker image
    Set-Location $InstallDir
    docker build -t ttrpg-assistant:latest .
    
    Write-Success "Docker setup complete"
}

# Create shortcuts
function Create-Shortcuts {
    Write-Header "Creating Shortcuts"
    
    # Create desktop shortcut
    $desktopPath = [Environment]::GetFolderPath("Desktop")
    $shortcutPath = "$desktopPath\TTRPG Assistant.lnk"
    
    $WshShell = New-Object -ComObject WScript.Shell
    $Shortcut = $WshShell.CreateShortcut($shortcutPath)
    $Shortcut.TargetPath = "powershell.exe"
    $Shortcut.Arguments = "-NoExit -Command `"& '$InstallDir\venv\Scripts\Activate.ps1'; python -m src.main`""
    $Shortcut.WorkingDirectory = $InstallDir
    $Shortcut.IconLocation = "$InstallDir\deploy\assets\icon.ico"
    $Shortcut.Description = "TTRPG Assistant MCP Server"
    $Shortcut.Save()
    
    Write-Success "Desktop shortcut created"
    
    # Create Start Menu shortcut
    $startMenuPath = [Environment]::GetFolderPath("StartMenu")
    $programsPath = "$startMenuPath\Programs\TTRPG Assistant"
    
    if (!(Test-Path $programsPath)) {
        New-Item -ItemType Directory -Path $programsPath -Force | Out-Null
    }
    
    $startShortcut = "$programsPath\TTRPG Assistant.lnk"
    $Shortcut = $WshShell.CreateShortcut($startShortcut)
    $Shortcut.TargetPath = "powershell.exe"
    $Shortcut.Arguments = "-NoExit -Command `"& '$InstallDir\venv\Scripts\Activate.ps1'; python -m src.main`""
    $Shortcut.WorkingDirectory = $InstallDir
    $Shortcut.IconLocation = "$InstallDir\deploy\assets\icon.ico"
    $Shortcut.Description = "TTRPG Assistant MCP Server"
    $Shortcut.Save()
    
    Write-Success "Start Menu shortcut created"
}

# Create uninstall script
function Create-UninstallScript {
    Write-Header "Creating Uninstall Script"
    
    $uninstallScript = @"
Write-Host "Uninstalling TTRPG Assistant MCP Server..." -ForegroundColor Yellow

# Stop service if installed
if (Get-Service -Name 'TTRPGAssistant' -ErrorAction SilentlyContinue) {
    Stop-Service -Name 'TTRPGAssistant' -Force
    & "$InstallDir\venv\Scripts\python.exe" "$InstallDir\service_wrapper.py" remove
}

# Backup data
`$backupPath = "`$env:TEMP\ttrpg-assistant-backup-`$(Get-Date -Format 'yyyyMMdd-HHmmss').zip"
Write-Host "Creating backup at `$backupPath"
Compress-Archive -Path "$DataDir" -DestinationPath `$backupPath

# Remove directories
Remove-Item -Path "$InstallDir" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$ConfigDir" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$LogDir" -Recurse -Force -ErrorAction SilentlyContinue

# Remove shortcuts
Remove-Item -Path "`$([Environment]::GetFolderPath('Desktop'))\TTRPG Assistant.lnk" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "`$([Environment]::GetFolderPath('StartMenu'))\Programs\TTRPG Assistant" -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "TTRPG Assistant MCP Server has been uninstalled." -ForegroundColor Green
Write-Host "Data backup saved to: `$backupPath" -ForegroundColor Cyan
Write-Host "To completely remove all data, delete: $DataDir" -ForegroundColor Yellow
"@
    
    Set-Content -Path "$InstallDir\uninstall.ps1" -Value $uninstallScript
    Write-Success "Uninstall script created at $InstallDir\uninstall.ps1"
}

# Run post-installation checks
function Run-PostInstallChecks {
    Write-Header "Running Post-Installation Checks"
    
    Set-Location $InstallDir
    & "$InstallDir\venv\Scripts\Activate.ps1"
    
    python deploy/scripts/check_requirements.py
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "All post-installation checks passed"
    } else {
        Write-Warning "Some checks failed, please review the output above"
    }
}

# Display completion message
function Show-CompletionMessage {
    Write-Header "Installation Complete!"
    
    Write-Host ""
    Write-Host "TTRPG Assistant MCP Server has been successfully installed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Installation Details:" -ForegroundColor Cyan
    Write-Host "  Install Directory: $InstallDir"
    Write-Host "  Data Directory: $DataDir"
    Write-Host "  Config Directory: $ConfigDir"
    Write-Host "  Log Directory: $LogDir"
    Write-Host ""
    
    switch ($InstallMode) {
        "service" {
            Write-Host "To start the service:" -ForegroundColor Yellow
            Write-Host "  Start-Service -Name 'TTRPGAssistant'"
            Write-Host ""
            Write-Host "To check service status:" -ForegroundColor Yellow
            Write-Host "  Get-Service -Name 'TTRPGAssistant'"
        }
        "docker" {
            Write-Host "To start with Docker:" -ForegroundColor Yellow
            Write-Host "  cd '$InstallDir'"
            Write-Host "  docker-compose up -d"
            Write-Host ""
            Write-Host "To view logs:" -ForegroundColor Yellow
            Write-Host "  docker-compose logs -f"
        }
        default {
            Write-Host "To start the server:" -ForegroundColor Yellow
            Write-Host "  Use the desktop shortcut or Start Menu shortcut"
            Write-Host ""
            Write-Host "Or run manually:" -ForegroundColor Yellow
            Write-Host "  cd '$InstallDir'"
            Write-Host "  & '.\venv\Scripts\Activate.ps1'"
            Write-Host "  python -m src.main"
        }
    }
    
    Write-Host ""
    Write-Host "Configuration files are located in: $ConfigDir" -ForegroundColor Cyan
    Write-Host "Please edit the .env file to customize your installation."
    Write-Host ""
    Write-Host "To uninstall, run:" -ForegroundColor Yellow
    Write-Host "  & '$InstallDir\uninstall.ps1'"
    Write-Host ""
}

# Main installation flow
try {
    # Check Python
    if (!(Check-Python)) {
        Install-Python
    }
    
    # Check Git
    Check-Git
    
    # Create directories
    Create-Directories
    
    # Setup repository
    Setup-Repository
    
    # Setup virtual environment
    Setup-VirtualEnvironment
    
    # Install dependencies
    Install-Dependencies
    
    # Configure application
    Configure-Application
    
    # Install Windows service (if requested)
    Install-WindowsService
    
    # Setup Docker (if requested)
    Setup-Docker
    
    # Create shortcuts
    Create-Shortcuts
    
    # Create uninstall script
    Create-UninstallScript
    
    # Run post-installation checks
    Run-PostInstallChecks
    
    # Show completion message
    Show-CompletionMessage
    
} catch {
    Write-Error "Installation failed: $_"
    exit 1
}