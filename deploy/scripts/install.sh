#!/bin/bash

#############################################################################
# TTRPG Assistant MCP Server - Installation Script for Linux/Mac
#############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Installation configuration
INSTALL_DIR="${INSTALL_DIR:-/opt/ttrpg-assistant}"
DATA_DIR="${DATA_DIR:-/var/lib/ttrpg-assistant}"
CONFIG_DIR="${CONFIG_DIR:-/etc/ttrpg-assistant}"
LOG_DIR="${LOG_DIR:-/var/log/ttrpg-assistant}"
SERVICE_USER="${SERVICE_USER:-ttrpg}"
PYTHON_VERSION="${PYTHON_VERSION:-3.9}"
INSTALL_MODE="${INSTALL_MODE:-standalone}"  # standalone, docker, systemd
GPU_SUPPORT="${GPU_SUPPORT:-none}"  # none, cuda, rocm

# Functions
print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_root() {
    if [[ $EUID -eq 0 ]]; then
        print_success "Running with root privileges"
    else
        print_error "This script must be run as root"
        echo "Please run: sudo $0"
        exit 1
    fi
}

detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            OS_TYPE="debian"
            PKG_MANAGER="apt-get"
        elif [ -f /etc/redhat-release ]; then
            OS_TYPE="redhat"
            PKG_MANAGER="yum"
        elif [ -f /etc/arch-release ]; then
            OS_TYPE="arch"
            PKG_MANAGER="pacman"
        else
            OS_TYPE="linux"
            PKG_MANAGER="unknown"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS_TYPE="macos"
        PKG_MANAGER="brew"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    print_success "Detected OS: $OS_TYPE (Package manager: $PKG_MANAGER)"
}

check_python() {
    print_header "Checking Python Installation"
    
    # Check for Python
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python not found. Installing Python ${PYTHON_VERSION}..."
        install_python
        return
    fi
    
    # Check Python version
    PYTHON_VERSION_INSTALLED=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    
    if [[ $(echo "$PYTHON_VERSION_INSTALLED >= $PYTHON_VERSION" | bc -l) -eq 1 ]]; then
        print_success "Python $PYTHON_VERSION_INSTALLED found"
    else
        print_warning "Python $PYTHON_VERSION_INSTALLED found, but $PYTHON_VERSION or higher is recommended"
    fi
}

install_python() {
    case $PKG_MANAGER in
        apt-get)
            apt-get update
            apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev
            ;;
        yum)
            yum install -y python${PYTHON_VERSION/./} python${PYTHON_VERSION/./}-devel
            ;;
        brew)
            brew install python@${PYTHON_VERSION}
            ;;
        pacman)
            pacman -S --noconfirm python python-pip
            ;;
        *)
            print_error "Cannot install Python automatically on this system"
            exit 1
            ;;
    esac
    
    PYTHON_CMD="python${PYTHON_VERSION}"
    print_success "Python ${PYTHON_VERSION} installed"
}

install_system_dependencies() {
    print_header "Installing System Dependencies"
    
    case $PKG_MANAGER in
        apt-get)
            apt-get update
            apt-get install -y \
                build-essential \
                curl \
                git \
                libmagic1 \
                libpq-dev \
                libssl-dev \
                libffi-dev \
                libxml2-dev \
                libxslt1-dev \
                libjpeg-dev \
                zlib1g-dev
            ;;
        yum)
            yum groupinstall -y "Development Tools"
            yum install -y \
                curl \
                git \
                file-libs \
                postgresql-devel \
                openssl-devel \
                libffi-devel \
                libxml2-devel \
                libxslt-devel \
                libjpeg-devel \
                zlib-devel
            ;;
        brew)
            brew install \
                curl \
                git \
                libmagic \
                postgresql \
                openssl \
                libffi \
                libxml2 \
                libxslt \
                jpeg \
                zlib
            ;;
        pacman)
            pacman -S --noconfirm \
                base-devel \
                curl \
                git \
                file \
                postgresql-libs \
                openssl \
                libffi \
                libxml2 \
                libxslt \
                libjpeg-turbo \
                zlib
            ;;
        *)
            print_warning "Cannot install system dependencies automatically"
            ;;
    esac
    
    print_success "System dependencies installed"
}

create_user() {
    print_header "Creating Service User"
    
    if id "$SERVICE_USER" &>/dev/null; then
        print_warning "User $SERVICE_USER already exists"
    else
        if [[ "$OS_TYPE" == "macos" ]]; then
            # macOS user creation
            dscl . -create /Users/$SERVICE_USER
            dscl . -create /Users/$SERVICE_USER UserShell /usr/bin/false
            dscl . -create /Users/$SERVICE_USER RealName "TTRPG Assistant Service"
            dscl . -create /Users/$SERVICE_USER UniqueID "510"
            dscl . -create /Users/$SERVICE_USER PrimaryGroupID 20
        else
            # Linux user creation
            useradd --system --shell /bin/false --home /nonexistent --no-create-home $SERVICE_USER
        fi
        print_success "Created user: $SERVICE_USER"
    fi
}

create_directories() {
    print_header "Creating Directory Structure"
    
    # Create directories
    mkdir -p "$INSTALL_DIR"
    mkdir -p "$DATA_DIR"/{chromadb,cache,backup,export}
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$LOG_DIR"
    
    # Set permissions
    chown -R $SERVICE_USER:$SERVICE_USER "$DATA_DIR"
    chown -R $SERVICE_USER:$SERVICE_USER "$LOG_DIR"
    chmod 755 "$INSTALL_DIR"
    chmod 750 "$DATA_DIR"
    chmod 750 "$CONFIG_DIR"
    chmod 750 "$LOG_DIR"
    
    print_success "Directory structure created"
}

clone_repository() {
    print_header "Cloning Repository"
    
    if [ -d "$INSTALL_DIR/.git" ]; then
        print_warning "Repository already exists, pulling latest changes"
        cd "$INSTALL_DIR"
        git pull origin main
    else
        print_warning "Copying from local directory instead of cloning"
        # Copy from current directory
        SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
        PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
        
        cp -r "$PROJECT_ROOT"/* "$INSTALL_DIR/" 2>/dev/null || true
        cp -r "$PROJECT_ROOT"/.* "$INSTALL_DIR/" 2>/dev/null || true
    fi
    
    chown -R $SERVICE_USER:$SERVICE_USER "$INSTALL_DIR"
    print_success "Repository prepared"
}

setup_virtual_environment() {
    print_header "Setting Up Python Virtual Environment"
    
    cd "$INSTALL_DIR"
    
    # Create virtual environment
    $PYTHON_CMD -m venv venv
    
    # Activate and upgrade pip
    source venv/bin/activate
    pip install --upgrade pip setuptools wheel
    
    print_success "Virtual environment created"
}

install_python_dependencies() {
    print_header "Installing Python Dependencies"
    
    cd "$INSTALL_DIR"
    source venv/bin/activate
    
    # Install based on GPU support
    case $GPU_SUPPORT in
        cuda)
            print_warning "Installing with CUDA support (this may take a while)"
            pip install torch torchvision torchaudio
            ;;
        rocm)
            print_warning "Installing with ROCm support (this may take a while)"
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
            ;;
        *)
            print_warning "Installing CPU-only version (no GPU support)"
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            ;;
    esac
    
    # Install main package
    pip install -e .
    
    # Download spaCy model
    python -m spacy download en_core_web_sm
    
    print_success "Python dependencies installed"
}

configure_application() {
    print_header "Configuring Application"
    
    # Copy configuration templates
    cp "$INSTALL_DIR/deploy/config/.env.template" "$CONFIG_DIR/.env"
    cp "$INSTALL_DIR/deploy/config/config.yaml.template" "$CONFIG_DIR/config.yaml"
    
    # Update configuration paths
    sed -i "s|CHROMA_DB_PATH=.*|CHROMA_DB_PATH=$DATA_DIR/chromadb|" "$CONFIG_DIR/.env"
    sed -i "s|CACHE_DIR=.*|CACHE_DIR=$DATA_DIR/cache|" "$CONFIG_DIR/.env"
    sed -i "s|LOG_FILE=.*|LOG_FILE=$LOG_DIR/ttrpg-assistant.log|" "$CONFIG_DIR/.env"
    sed -i "s|SECURITY_LOG_FILE=.*|SECURITY_LOG_FILE=$LOG_DIR/security.log|" "$CONFIG_DIR/.env"
    
    # Set proper permissions
    chmod 640 "$CONFIG_DIR/.env"
    chmod 640 "$CONFIG_DIR/config.yaml"
    chown $SERVICE_USER:$SERVICE_USER "$CONFIG_DIR"/*
    
    print_success "Application configured"
}

install_systemd_service() {
    if [[ "$INSTALL_MODE" != "systemd" ]]; then
        return
    fi
    
    print_header "Installing systemd Service"
    
    if [[ "$OS_TYPE" == "macos" ]]; then
        print_warning "systemd is not available on macOS, skipping service installation"
        return
    fi
    
    # Create service file from template
    cat "$INSTALL_DIR/deploy/config/systemd.service.template" | \
        sed "s|{{INSTALL_DIR}}|$INSTALL_DIR|g" | \
        sed "s|{{SERVICE_USER}}|$SERVICE_USER|g" | \
        sed "s|{{CONFIG_DIR}}|$CONFIG_DIR|g" > /etc/systemd/system/ttrpg-assistant.service
    
    # Reload systemd and enable service
    systemctl daemon-reload
    systemctl enable ttrpg-assistant.service
    
    print_success "systemd service installed and enabled"
}

setup_docker() {
    if [[ "$INSTALL_MODE" != "docker" ]]; then
        return
    fi
    
    print_header "Setting Up Docker"
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Copy Docker Compose template
    cp "$INSTALL_DIR/deploy/config/docker-compose.yaml.template" "$INSTALL_DIR/docker-compose.yaml"
    
    # Update paths in docker-compose.yaml
    sed -i "s|{{DATA_DIR}}|$DATA_DIR|g" "$INSTALL_DIR/docker-compose.yaml"
    sed -i "s|{{CONFIG_DIR}}|$CONFIG_DIR|g" "$INSTALL_DIR/docker-compose.yaml"
    sed -i "s|{{LOG_DIR}}|$LOG_DIR|g" "$INSTALL_DIR/docker-compose.yaml"
    
    # Build Docker image
    cd "$INSTALL_DIR"
    docker build -t ttrpg-assistant:latest .
    
    print_success "Docker setup complete"
}

run_post_install_checks() {
    print_header "Running Post-Installation Checks"
    
    cd "$INSTALL_DIR"
    source venv/bin/activate
    
    # Run basic checks
    python deploy/scripts/check_requirements.py
    
    if [ $? -eq 0 ]; then
        print_success "All post-installation checks passed"
    else
        print_warning "Some checks failed, please review the output above"
    fi
}

create_uninstall_script() {
    print_header "Creating Uninstall Script"
    
    cat > "$INSTALL_DIR/uninstall.sh" << 'EOF'
#!/bin/bash

echo "Uninstalling TTRPG Assistant MCP Server..."

# Stop service if running
if systemctl is-active --quiet ttrpg-assistant; then
    systemctl stop ttrpg-assistant
    systemctl disable ttrpg-assistant
fi

# Remove service file
rm -f /etc/systemd/system/ttrpg-assistant.service

# Backup data before removal
if [ -d "/var/lib/ttrpg-assistant" ]; then
    echo "Creating backup of data..."
    tar czf /tmp/ttrpg-assistant-backup-$(date +%Y%m%d-%H%M%S).tar.gz /var/lib/ttrpg-assistant
    echo "Backup saved to /tmp/"
fi

# Remove directories
rm -rf /opt/ttrpg-assistant
rm -rf /etc/ttrpg-assistant
rm -rf /var/log/ttrpg-assistant

# Optional: Remove data directory (commented by default for safety)
# rm -rf /var/lib/ttrpg-assistant

# Remove user
if id "ttrpg" &>/dev/null; then
    userdel ttrpg
fi

echo "TTRPG Assistant MCP Server has been uninstalled."
echo "Data directory /var/lib/ttrpg-assistant has been preserved."
echo "To completely remove all data, run: rm -rf /var/lib/ttrpg-assistant"
EOF
    
    chmod +x "$INSTALL_DIR/uninstall.sh"
    print_success "Uninstall script created at $INSTALL_DIR/uninstall.sh"
}

print_completion_message() {
    print_header "Installation Complete!"
    
    echo ""
    echo "TTRPG Assistant MCP Server has been successfully installed!"
    echo ""
    echo "Installation Details:"
    echo "  Install Directory: $INSTALL_DIR"
    echo "  Data Directory: $DATA_DIR"
    echo "  Config Directory: $CONFIG_DIR"
    echo "  Log Directory: $LOG_DIR"
    echo "  Service User: $SERVICE_USER"
    echo ""
    
    case $INSTALL_MODE in
        systemd)
            echo "To start the service:"
            echo "  systemctl start ttrpg-assistant"
            echo ""
            echo "To check service status:"
            echo "  systemctl status ttrpg-assistant"
            echo ""
            echo "To view logs:"
            echo "  journalctl -u ttrpg-assistant -f"
            ;;
        docker)
            echo "To start with Docker:"
            echo "  cd $INSTALL_DIR"
            echo "  docker-compose up -d"
            echo ""
            echo "To view logs:"
            echo "  docker-compose logs -f"
            ;;
        standalone)
            echo "To start the server:"
            echo "  cd $INSTALL_DIR"
            echo "  source venv/bin/activate"
            echo "  python -m src.main"
            ;;
    esac
    
    echo ""
    echo "Configuration files are located in: $CONFIG_DIR"
    echo "Please edit the .env file to customize your installation."
    echo ""
    echo "To uninstall, run: $INSTALL_DIR/uninstall.sh"
    echo ""
}

# Main installation flow
main() {
    print_header "TTRPG Assistant MCP Server Installation"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --install-dir)
                INSTALL_DIR="$2"
                shift 2
                ;;
            --data-dir)
                DATA_DIR="$2"
                shift 2
                ;;
            --config-dir)
                CONFIG_DIR="$2"
                shift 2
                ;;
            --log-dir)
                LOG_DIR="$2"
                shift 2
                ;;
            --user)
                SERVICE_USER="$2"
                shift 2
                ;;
            --mode)
                INSTALL_MODE="$2"
                shift 2
                ;;
            --gpu)
                GPU_SUPPORT="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  --install-dir DIR    Installation directory (default: /opt/ttrpg-assistant)"
                echo "  --data-dir DIR       Data directory (default: /var/lib/ttrpg-assistant)"
                echo "  --config-dir DIR     Configuration directory (default: /etc/ttrpg-assistant)"
                echo "  --log-dir DIR        Log directory (default: /var/log/ttrpg-assistant)"
                echo "  --user USER          Service user (default: ttrpg)"
                echo "  --mode MODE          Installation mode: standalone, systemd, docker (default: standalone)"
                echo "  --gpu SUPPORT        GPU support: none, cuda, rocm (default: none)"
                echo "  --help               Show this help message"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                echo "Run $0 --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Check for root privileges
    check_root
    
    # Detect operating system
    detect_os
    
    # Run installation steps
    check_python
    install_system_dependencies
    create_user
    create_directories
    clone_repository
    setup_virtual_environment
    install_python_dependencies
    configure_application
    install_systemd_service
    setup_docker
    run_post_install_checks
    create_uninstall_script
    print_completion_message
}

# Run main installation
main "$@"