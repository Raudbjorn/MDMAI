#!/bin/bash

# TTRPG Assistant - Quick Setup Script
# This script helps set up the development environment

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════╗"
echo "║     TTRPG Assistant MCP Server - Quick Setup              ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print colored output
print_status() {
    echo "▶ $1"
}

print_success() {
    echo "✅ $1"
}

print_error() {
    echo "❌ $1"
}

print_warning() {
    echo "⚠️  $1"
}

# Detect the package manager preference
PACKAGE_MANAGER=""

if command_exists uv; then
    PACKAGE_MANAGER="uv"
    print_success "Found uv (fast, modern package manager)"
elif command_exists poetry; then
    PACKAGE_MANAGER="poetry"
    print_success "Found Poetry package manager"
else
    print_warning "Neither uv nor Poetry found. Would you like to install one?"
    echo ""
    echo "1) Install uv (recommended - faster)"
    echo "2) Install Poetry (more established)"
    echo "3) Use pip with virtual environment"
    echo "4) Exit"
    echo ""
    read -p "Choose an option [1-4]: " choice
    
    case $choice in
        1)
            print_status "Installing uv..."
            curl -LsSf https://astral.sh/uv/install.sh | sh
            export PATH="$HOME/.cargo/bin:$PATH"
            PACKAGE_MANAGER="uv"
            print_success "uv installed successfully!"
            ;;
        2)
            print_status "Installing Poetry..."
            curl -sSL https://install.python-poetry.org | python3 -
            export PATH="$HOME/.local/bin:$PATH"
            PACKAGE_MANAGER="poetry"
            print_success "Poetry installed successfully!"
            ;;
        3)
            PACKAGE_MANAGER="pip"
            print_status "Using pip with virtual environment"
            ;;
        4)
            print_status "Exiting setup"
            exit 0
            ;;
        *)
            print_error "Invalid option"
            exit 1
            ;;
    esac
fi

echo ""

# Detect GPU hardware
GPU_TYPE="cpu"
if command_exists nvidia-smi; then
    if nvidia-smi &>/dev/null; then
        GPU_TYPE="cuda"
        print_status "Detected NVIDIA GPU"
    fi
elif command_exists rocm-smi; then
    if rocm-smi &>/dev/null; then
        GPU_TYPE="rocm"
        print_status "Detected AMD GPU (ROCm)"
    fi
elif [ -d "/sys/class/drm" ] && ls /sys/class/drm/card*/device/vendor 2>/dev/null | xargs cat 2>/dev/null | grep -q "0x1002\|0x8086"; then
    # Check for AMD (0x1002) or Intel (0x8086) GPU
    if ls /sys/class/drm/card*/device/vendor 2>/dev/null | xargs cat 2>/dev/null | grep -q "0x1002"; then
        print_warning "AMD GPU detected but ROCm not installed. Using CPU-only mode."
        print_status "For GPU support, install ROCm and run 'make install-rocm'"
    fi
else
    print_status "No GPU detected, using CPU-only installation"
fi

echo ""
print_status "Setting up TTRPG Assistant with $PACKAGE_MANAGER..."
echo ""

# Setup based on package manager
case $PACKAGE_MANAGER in
    uv)
        # Check if venv already exists
        if [ -d ".venv" ]; then
            print_status "Virtual environment already exists, using existing environment..."
        else
            print_status "Creating virtual environment with uv..."
            uv venv
        fi
        
        print_status "Installing/updating dependencies..."
        uv pip install -e ".[dev,test,docs]" 2>&1 | grep -v "warning: A virtual environment already exists" || true
        
        # Install appropriate PyTorch version
        if [ "$GPU_TYPE" = "cuda" ]; then
            print_status "Installing CUDA-enabled PyTorch (this may take a while)..."
            uv pip install torch 2>&1 | grep -v "warning:" || true
        elif [ "$GPU_TYPE" = "rocm" ]; then
            print_status "Installing ROCm-enabled PyTorch..."
            uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.2 2>&1 | grep -v "warning:" || true
        else
            print_status "Installing CPU-only PyTorch (smaller download)..."
            uv pip install torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | grep -v "warning:" || true
        fi
        
        print_success "Setup complete with uv!"
        echo ""
        echo "To activate the environment, run:"
        echo "  source .venv/bin/activate"
        ;;
    
    poetry)
        print_status "Installing dependencies with Poetry..."
        poetry install --with dev,test,docs
        
        print_success "Setup complete with Poetry!"
        echo ""
        echo "To activate the environment, run:"
        echo "  poetry shell"
        ;;
    
    pip)
        # Check if venv already exists
        if [ -d ".venv" ]; then
            print_status "Virtual environment already exists, using existing environment..."
        else
            print_status "Creating virtual environment..."
            python3 -m venv .venv
        fi
        
        print_status "Activating virtual environment..."
        source .venv/bin/activate
        
        print_status "Upgrading pip..."
        pip install --upgrade pip --quiet
        
        print_status "Installing/updating dependencies..."
        pip install -e ".[dev,test,docs]" --quiet --upgrade
        
        # Install appropriate PyTorch version
        if [ "$GPU_TYPE" = "cuda" ]; then
            print_status "Installing CUDA-enabled PyTorch (this may take a while)..."
            pip install torch --quiet
        elif [ "$GPU_TYPE" = "rocm" ]; then
            print_status "Installing ROCm-enabled PyTorch..."
            pip install torch --index-url https://download.pytorch.org/whl/rocm6.2 --quiet
        else
            print_status "Installing CPU-only PyTorch (smaller download)..."
            pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
        fi
        
        print_success "Setup complete with pip!"
        echo ""
        echo "To activate the environment, run:"
        echo "  source .venv/bin/activate"
        ;;
esac

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                    Setup Complete! 🎉                     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Available commands (after activating environment):"
echo ""
echo "  make help         - Show all available commands"
echo "  make test         - Run tests with coverage"
echo "  make lint         - Run code quality checks"
echo "  make format       - Format code automatically"
echo "  make run          - Start the MCP server"
echo ""
echo "Quick start:"
echo "  1. Activate environment (see above)"
echo "  2. Run 'make test' to verify setup"
echo "  3. Run 'make run' to start the server"
echo ""
print_success "Happy gaming! 🎲"