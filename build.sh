#!/bin/bash

# TTRPG Assistant - Unified Build Script
# Combines all build systems: Python backend, SvelteKit webapp, Tauri desktop, PyOxidizer
# Automatically detects package managers and handles dependency management

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Emojis for better UX
ROCKET="🚀"
CHECK="✅"
CROSS="❌"
WARNING="⚠️"
GEAR="⚙️"
PACKAGE="📦"
TEST="🧪"
CLEAN="🧹"

# Print functions
print_header() {
    echo -e "\n${PURPLE}╔═══════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${PURPLE}║                        TTRPG Assistant Build System                           ║${NC}"
    echo -e "${PURPLE}╚═══════════════════════════════════════════════════════════════════════════════╝${NC}"
}

print_section() {
    echo -e "\n${CYAN}${GEAR} $1${NC}"
    echo -e "${CYAN}$(printf '%.0s─' {1..80})${NC}"
}

print_success() {
    echo -e "${GREEN}${CHECK} $1${NC}"
}

print_error() {
    echo -e "${RED}${CROSS} $1${NC}" >&2
}

print_warning() {
    echo -e "${YELLOW}${WARNING} $1${NC}"
}

print_info() {
    echo -e "${BLUE}${GEAR} $1${NC}"
}

# Command existence check
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Directory detection
get_script_dir() {
    cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd
}

# Get project root
PROJECT_ROOT=$(get_script_dir)
cd "$PROJECT_ROOT"

# Package manager detection
detect_python_manager() {
    if command_exists uv; then
        echo "uv"
    elif command_exists poetry; then
        echo "poetry"
    else
        echo "pip"
    fi
}

detect_node_manager() {
    if command_exists pnpm; then
        echo "pnpm"
    elif command_exists yarn; then
        echo "yarn"
    elif command_exists npm; then
        echo "npm"
    else
        echo "none"
    fi
}

# Setup Python virtual environment
setup_python_env() {
    local manager=$1
    print_info "Setting up Python environment with $manager"
    
    case $manager in
        uv)
            if [ ! -d ".venv" ]; then
                print_info "Creating virtual environment with uv..."
                uv venv
            fi
            print_info "Installing Python dependencies..."
            uv pip install -e ".[dev,test,docs]" --quiet
            ;;
        poetry)
            print_info "Installing Python dependencies with Poetry..."
            poetry install --with dev,test,docs --quiet
            ;;
        pip)
            if [ ! -d ".venv" ]; then
                print_info "Creating virtual environment..."
                python3 -m venv .venv
            fi
            source .venv/bin/activate
            print_info "Installing Python dependencies with pip..."
            pip install --upgrade pip --quiet
            pip install -e ".[dev,test,docs]" --quiet
            ;;
    esac
}

# Setup Node.js dependencies
setup_node_env() {
    local manager=$1
    local dir=$2
    
    if [ ! -d "$dir" ]; then
        print_warning "Directory $dir not found, skipping Node.js setup"
        return
    fi
    
    cd "$dir"
    print_info "Setting up Node.js environment in $dir with $manager"
    
    case $manager in
        pnpm)
            pnpm install --frozen-lockfile --silent
            ;;
        yarn)
            yarn install --frozen-lockfile --silent
            ;;
        npm)
            npm install --silent
            ;;
        none)
            print_error "No Node.js package manager found"
            return 1
            ;;
    esac
    
    cd "$PROJECT_ROOT"
}

# Run Python command with detected manager
run_python_cmd() {
    local cmd="$1"
    local manager=$(detect_python_manager)
    
    case $manager in
        uv)
            if [ -f ".venv/bin/activate" ]; then
                source .venv/bin/activate && eval "$cmd"
            else
                uv run $cmd
            fi
            ;;
        poetry)
            poetry run $cmd
            ;;
        pip)
            if [ -f ".venv/bin/activate" ]; then
                source .venv/bin/activate && eval "$cmd"
            else
                eval "$cmd"
            fi
            ;;
    esac
}

# Run Node.js command with detected manager
run_node_cmd() {
    local cmd="$1"
    local dir="$2"
    local manager=$(detect_node_manager)
    
    if [ ! -d "$dir" ]; then
        print_warning "Directory $dir not found, skipping command: $cmd"
        return
    fi
    
    cd "$dir"
    
    case $manager in
        pnpm)
            pnpm $cmd
            ;;
        yarn)
            yarn $cmd
            ;;
        npm)
            npm run $cmd
            ;;
        none)
            print_error "No Node.js package manager found"
            return 1
            ;;
    esac
    
    cd "$PROJECT_ROOT"
}

# Install dependencies
install_dependencies() {
    print_section "Installing Dependencies"
    
    local python_manager=$(detect_python_manager)
    local node_manager=$(detect_node_manager)
    
    print_info "Detected Python manager: $python_manager"
    print_info "Detected Node.js manager: $node_manager"
    
    # Python dependencies
    setup_python_env "$python_manager"
    print_success "Python dependencies installed"
    
    # Frontend dependencies
    setup_node_env "$node_manager" "frontend"
    print_success "Frontend dependencies installed"
    
    # Desktop frontend dependencies
    setup_node_env "$node_manager" "desktop/frontend"
    print_success "Desktop frontend dependencies installed"
    
    # Rust/Cargo dependencies (handled by Cargo automatically)
    if command_exists cargo; then
        print_info "Cargo detected - Rust dependencies will be handled automatically"
    else
        print_warning "Cargo not found - desktop builds will not be available"
    fi
}

# Build components
build_backend() {
    print_section "Building Python Backend"
    
    print_info "Running Python type checks..."
    run_python_cmd "mypy src --ignore-missing-imports"
    print_success "Type checks passed"
    
    print_info "Running Python linting..."
    run_python_cmd "flake8 src tests --max-line-length=100"
    print_success "Linting passed"
    
    print_success "Python backend ready"
}

build_webapp() {
    print_section "Building Web Application"
    
    if [ ! -d "frontend" ]; then
        print_warning "Frontend directory not found, skipping webapp build"
        return
    fi
    
    print_info "Type checking TypeScript..."
    run_node_cmd "check" "frontend"
    print_success "TypeScript checks passed"
    
    print_info "Linting frontend code..."
    run_node_cmd "lint" "frontend"
    print_success "Frontend linting passed"
    
    print_info "Building webapp for production..."
    run_node_cmd "build" "frontend"
    print_success "Web application built successfully"
}

build_desktop() {
    print_section "Building Desktop Application"
    
    if [ ! -d "desktop/frontend" ]; then
        print_warning "Desktop frontend directory not found, skipping desktop build"
        return
    fi
    
    if ! command_exists cargo; then
        print_error "Cargo not found - cannot build desktop application"
        return 1
    fi
    
    print_info "Type checking desktop TypeScript..."
    run_node_cmd "check" "desktop/frontend"
    print_success "Desktop TypeScript checks passed"
    
    print_info "Linting desktop frontend..."
    run_node_cmd "lint" "desktop/frontend"
    print_success "Desktop frontend linting passed"
    
    print_info "Building desktop frontend..."
    run_node_cmd "build" "desktop/frontend"
    print_success "Desktop frontend built"
    
    print_info "Building Rust/Tauri application..."
    cd desktop/frontend
    cargo check --manifest-path src-tauri/Cargo.toml
    print_success "Rust compilation check passed"
    cd "$PROJECT_ROOT"
    
    print_success "Desktop application ready for development"
}

build_desktop_release() {
    print_section "Building Desktop Application (Release)"
    
    if [ ! -d "desktop/frontend" ]; then
        print_error "Desktop frontend directory not found"
        return 1
    fi
    
    if ! command_exists cargo; then
        print_error "Cargo not found - cannot build desktop application"
        return 1
    fi
    
    print_info "Building desktop application for release..."
    run_node_cmd "tauri:build" "desktop/frontend"
    print_success "Desktop application built for release"
}

# Testing
run_tests() {
    print_section "Running Tests"
    
    # Python tests
    print_info "Running Python tests..."
    run_python_cmd "pytest -v"
    print_success "Python tests passed"
    
    # Frontend tests (if available)
    if [ -d "frontend" ]; then
        print_info "Running frontend tests..."
        run_node_cmd "test" "frontend" 2>/dev/null || print_warning "Frontend tests not available or failed"
    fi
    
    # Desktop frontend tests (if available)
    if [ -d "desktop/frontend" ]; then
        print_info "Checking desktop TypeScript..."
        run_node_cmd "check" "desktop/frontend"
        print_success "Desktop TypeScript validation passed"
    fi
    
    print_success "All tests completed"
}

# Linting and formatting
lint_all() {
    print_section "Linting All Code"
    
    # Python linting
    print_info "Linting Python code..."
    run_python_cmd "flake8 src tests --max-line-length=100" && print_success "Python linting passed" || print_error "Python linting failed"
    run_python_cmd "mypy src --ignore-missing-imports" && print_success "Python type checking passed" || print_error "Python type checking failed"
    
    # Frontend linting
    if [ -d "frontend" ]; then
        print_info "Linting frontend..."
        run_node_cmd "lint" "frontend" && print_success "Frontend linting passed" || print_error "Frontend linting failed"
    fi
    
    # Desktop frontend linting
    if [ -d "desktop/frontend" ]; then
        print_info "Linting desktop frontend..."
        run_node_cmd "lint" "desktop/frontend" && print_success "Desktop frontend linting passed" || print_error "Desktop frontend linting failed"
    fi
}

format_all() {
    print_section "Formatting All Code"
    
    # Python formatting
    print_info "Formatting Python code..."
    run_python_cmd "black src tests"
    run_python_cmd "isort src tests"
    print_success "Python code formatted"
    
    # Frontend formatting
    if [ -d "frontend" ]; then
        print_info "Formatting frontend..."
        run_node_cmd "format" "frontend"
        print_success "Frontend code formatted"
    fi
    
    # Desktop frontend formatting
    if [ -d "desktop/frontend" ]; then
        print_info "Formatting desktop frontend..."
        run_node_cmd "format" "desktop/frontend"
        print_success "Desktop frontend code formatted"
    fi
}

# Development servers
dev_backend() {
    print_section "Starting Backend Development Server"
    run_python_cmd "python -m src.main"
}

dev_webapp() {
    print_section "Starting Web Application Development Server"
    
    if [ ! -d "frontend" ]; then
        print_error "Frontend directory not found"
        return 1
    fi
    
    run_node_cmd "dev" "frontend"
}

dev_desktop() {
    print_section "Starting Desktop Application Development"
    
    if [ ! -d "desktop/frontend" ]; then
        print_error "Desktop frontend directory not found"
        return 1
    fi
    
    run_node_cmd "tauri:dev" "desktop/frontend"
}

# Cleaning
clean_all() {
    print_section "Cleaning Build Artifacts"
    
    print_info "Cleaning Python artifacts..."
    rm -rf build dist *.egg-info
    rm -rf .pytest_cache .coverage htmlcov
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete
    
    print_info "Cleaning Node.js artifacts..."
    [ -d "frontend/node_modules" ] && rm -rf frontend/node_modules
    [ -d "frontend/build" ] && rm -rf frontend/build
    [ -d "frontend/.svelte-kit" ] && rm -rf frontend/.svelte-kit
    [ -d "desktop/frontend/node_modules" ] && rm -rf desktop/frontend/node_modules
    [ -d "desktop/frontend/build" ] && rm -rf desktop/frontend/build
    [ -d "desktop/frontend/.svelte-kit" ] && rm -rf desktop/frontend/.svelte-kit
    
    print_info "Cleaning Rust artifacts..."
    [ -d "desktop/frontend/src-tauri/target" ] && rm -rf desktop/frontend/src-tauri/target
    
    print_success "All artifacts cleaned"
}

# Help/Usage
show_help() {
    print_header
    echo -e "\n${BLUE}Usage: $0 [command]${NC}\n"
    
    echo -e "${YELLOW}Setup Commands:${NC}"
    echo -e "  ${GREEN}deps${NC}              Install all dependencies (auto-detects package managers)"
    echo -e "  ${GREEN}setup${NC}             Alias for deps"
    echo ""
    
    echo -e "${YELLOW}Build Commands:${NC}"
    echo -e "  ${GREEN}build${NC}             Build all components (backend + webapp + desktop)"
    echo -e "  ${GREEN}backend${NC}           Build and validate Python backend only"
    echo -e "  ${GREEN}webapp${NC}            Build SvelteKit web application only"
    echo -e "  ${GREEN}desktop${NC}           Build desktop application (development)"
    echo -e "  ${GREEN}desktop-release${NC}   Build desktop application (release/production)"
    echo ""
    
    echo -e "${YELLOW}Development Commands:${NC}"
    echo -e "  ${GREEN}dev-backend${NC}       Start Python MCP server in development mode"
    echo -e "  ${GREEN}dev-webapp${NC}        Start SvelteKit development server"
    echo -e "  ${GREEN}dev-desktop${NC}       Start Tauri desktop application in development"
    echo ""
    
    echo -e "${YELLOW}Quality Assurance:${NC}"
    echo -e "  ${GREEN}test${NC}              Run all tests (Python + TypeScript)"
    echo -e "  ${GREEN}lint${NC}              Run linting on all code"
    echo -e "  ${GREEN}format${NC}            Format all code (Python + TypeScript)"
    echo ""
    
    echo -e "${YELLOW}Utility Commands:${NC}"
    echo -e "  ${GREEN}clean${NC}             Remove all build artifacts and caches"
    echo -e "  ${GREEN}help${NC}              Show this help message"
    echo ""
    
    echo -e "${YELLOW}Examples:${NC}"
    echo -e "  ${CYAN}$0 deps && $0 build${NC}     # Full setup and build"
    echo -e "  ${CYAN}$0 dev-desktop${NC}          # Start desktop development"
    echo -e "  ${CYAN}$0 test && $0 lint${NC}      # Quality assurance pipeline"
    echo -e "  ${CYAN}$0 clean && $0 setup${NC}    # Clean rebuild"
    echo ""
    
    echo -e "${YELLOW}Detected Tools:${NC}"
    echo -e "  Python Manager: ${CYAN}$(detect_python_manager)${NC}"
    echo -e "  Node.js Manager: ${CYAN}$(detect_node_manager)${NC}"
    echo -e "  Rust/Cargo: ${CYAN}$(command_exists cargo && echo "available" || echo "not found")${NC}"
    echo ""
}

# Main command dispatcher
case "${1:-help}" in
    "deps"|"setup")
        print_header
        install_dependencies
        print_success "Setup complete! Run '$0 build' to build all components"
        ;;
    
    "build")
        print_header
        build_backend
        build_webapp
        build_desktop
        print_success "All components built successfully!"
        ;;
    
    "backend")
        print_header
        build_backend
        ;;
    
    "webapp")
        print_header
        build_webapp
        ;;
    
    "desktop")
        print_header
        build_desktop
        ;;
    
    "desktop-release")
        print_header
        build_desktop_release
        ;;
    
    "test")
        print_header
        run_tests
        ;;
    
    "lint")
        print_header
        lint_all
        ;;
    
    "format")
        print_header
        format_all
        ;;
    
    "dev-backend")
        print_header
        dev_backend
        ;;
    
    "dev-webapp")
        print_header
        dev_webapp
        ;;
    
    "dev-desktop")
        print_header
        dev_desktop
        ;;
    
    "clean")
        print_header
        clean_all
        ;;
    
    "help"|"-h"|"--help")
        show_help
        ;;
    
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac