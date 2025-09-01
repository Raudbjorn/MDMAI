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
    
    # Show git/GitHub status warnings
    check_git_status
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

# Git repository status check
check_git_status() {
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        return 0  # Not a git repo, skip checks
    fi
    
    local warnings=()
    
    # Check for uncommitted changes
    local uncommitted=$(git status --porcelain 2>/dev/null | wc -l)
    if [ "$uncommitted" -gt 20 ]; then
        warnings+=("🔄 You have $uncommitted uncommitted changes - consider committing or stashing")
    elif [ "$uncommitted" -gt 5 ]; then
        warnings+=("📝 You have $uncommitted uncommitted changes")
    fi
    
    # Check for unpushed commits and branch divergence
    local current_branch=$(git branch --show-current 2>/dev/null || echo "")
    if [ -n "$current_branch" ]; then
        local unpushed=$(git rev-list --count @{u}..HEAD 2>/dev/null || echo "0")
        if [ "$unpushed" -gt 0 ]; then
            warnings+=("📤 You have $unpushed unpushed commits on branch '$current_branch'")
        fi
        
        # Check if branch can be merged with main/master (quick check)
        check_merge_status warnings "$current_branch"
        
        # Check if branch is significantly behind main/master
        check_branch_divergence warnings "$current_branch"
    fi
    
    # Check for unmerged pull requests (if gh CLI is available)
    if command_exists gh; then
        check_github_status warnings
    fi
    
    # Display warnings if any
    if [ ${#warnings[@]} -gt 0 ]; then
        echo -e "\n${YELLOW}${WARNING} Git Status Notifications:${NC}"
        for warning in "${warnings[@]}"; do
            echo -e "  ${YELLOW}$warning${NC}"
        done
        echo ""
    fi
}

# Check if current branch can merge cleanly with main/master
check_merge_status() {
    local -n warnings_ref=$1
    local current_branch=$2
    
    # Skip if we're on main/master
    if [[ "$current_branch" == "main" || "$current_branch" == "master" ]]; then
        return 0
    fi
    
    # Find the default branch (main or master)
    local default_branch=""
    if git show-ref --verify --quiet refs/heads/main; then
        default_branch="main"
    elif git show-ref --verify --quiet refs/heads/master; then
        default_branch="master"
    else
        return 0  # No default branch found
    fi
    
    # Quick merge conflict check (this is fast)
    local merge_base=$(git merge-base "$current_branch" "$default_branch" 2>/dev/null || echo "")
    if [ -n "$merge_base" ]; then
        # Check if there are conflicting files (this is the expensive part, so we limit it)
        local conflicts=$(git merge-tree "$merge_base" "$current_branch" "$default_branch" 2>/dev/null | grep -c "<<<<<<< " 2>/dev/null || echo "0")
        if [ -n "$conflicts" ] && [ "$conflicts" -gt 0 ] 2>/dev/null; then
            warnings_ref+=("⚠️ Branch '$current_branch' may have merge conflicts with '$default_branch' ($conflicts potential conflicts)")
        fi
    fi
}

# Check branch divergence from main/master
check_branch_divergence() {
    local -n warnings_ref=$1
    local current_branch=$2
    
    # Skip if we're on main/master
    if [[ "$current_branch" == "main" || "$current_branch" == "master" ]]; then
        return 0
    fi
    
    # Find the default branch
    local default_branch=""
    if git show-ref --verify --quiet refs/heads/main; then
        default_branch="main"
    elif git show-ref --verify --quiet refs/heads/master; then
        default_branch="master"
    else
        return 0
    fi
    
    # Check how far behind we are (this is very fast)
    local behind=$(git rev-list --count HEAD.."$default_branch" 2>/dev/null || echo "0")
    local ahead=$(git rev-list --count "$default_branch"..HEAD 2>/dev/null || echo "0")
    
    if [ "$behind" -gt 20 ]; then
        warnings_ref+=("📉 Branch '$current_branch' is $behind commits behind '$default_branch' - consider rebasing")
    elif [ "$behind" -gt 5 ]; then
        warnings_ref+=("📋 Branch '$current_branch' is $behind commits behind '$default_branch'")
    fi
    
    # Check for very long-running branches
    local days_old=$(git log --format="%ct" -1 "$default_branch" 2>/dev/null)
    local branch_base=$(git merge-base "$current_branch" "$default_branch" 2>/dev/null)
    if [ -n "$days_old" ] && [ -n "$branch_base" ]; then
        local base_time=$(git log --format="%ct" -1 "$branch_base" 2>/dev/null || echo "$days_old")
        local days_since=$(( ($(date +%s) - base_time) / 86400 ))
        if [ "$days_since" -gt 30 ]; then
            warnings_ref+=("📅 Branch '$current_branch' diverged $days_since days ago - consider updating or merging")
        fi
    fi
}

# GitHub CLI integration for PR checks
check_github_status() {
    local -n warnings_ref=$1
    
    # Check if we're in a GitHub repo
    local github_repo=$(gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null || echo "")
    if [ -z "$github_repo" ]; then
        return 0  # Not a GitHub repo or not authenticated
    fi
    
    # Check for open pull requests with detailed status
    local open_prs=$(gh pr list --state open --json number,title,author,isDraft,mergeable,reviewDecision 2>/dev/null)
    local pr_count=$(echo "$open_prs" | jq length 2>/dev/null || echo "0")
    
    if [ "$pr_count" -gt 0 ]; then
        # Count mergeable vs problematic PRs
        local mergeable_count=$(echo "$open_prs" | jq '[.[] | select(.mergeable == "MERGEABLE")] | length' 2>/dev/null || echo "0")
        local conflicted_count=$(echo "$open_prs" | jq '[.[] | select(.mergeable == "CONFLICTING")] | length' 2>/dev/null || echo "0")
        local draft_count=$(echo "$open_prs" | jq '[.[] | select(.isDraft == true)] | length' 2>/dev/null || echo "0")
        
        warnings_ref+=("🔀 There are $pr_count open pull request(s) in $github_repo")
        
        if [ "$conflicted_count" -gt 0 ]; then
            warnings_ref+=("⚠️ $conflicted_count PR(s) have merge conflicts")
        fi
        
        if [ "$draft_count" -gt 0 ]; then
            warnings_ref+=("📝 $draft_count draft PR(s) not ready for review")
        fi
        
        # Show specific PRs if not too many
        if [ "$pr_count" -le 5 ]; then
            local pr_info=$(echo "$open_prs" | jq -r '.[] | "  • #\(.number): \(.title) (@\(.author.login))" + (if .isDraft then " [DRAFT]" else "" end) + (if .mergeable == "CONFLICTING" then " [CONFLICTS]" else "" end)' 2>/dev/null | head -3)
            if [ -n "$pr_info" ]; then
                warnings_ref+=("$pr_info")
            fi
        fi
    fi
    
    # Check for PRs that need review (assigned to you)
    local review_prs=$(gh pr list --state open --review-requested @me --json number 2>/dev/null | jq length 2>/dev/null || echo "0")
    if [ -n "$review_prs" ] && [ "$review_prs" -gt 0 ] 2>/dev/null; then
        warnings_ref+=("👀 You have $review_prs pull request(s) awaiting your review")
    fi
    
    # Check for failed CI/CD runs on current branch
    local current_branch=$(git branch --show-current 2>/dev/null || echo "")
    if [ -n "$current_branch" ]; then
        local failed_runs=$(gh run list --branch "$current_branch" --status failure --limit 5 --json conclusion 2>/dev/null | jq length 2>/dev/null || echo "0")
        if [ "$failed_runs" -gt 0 ]; then
            warnings_ref+=("❌ Recent CI/CD failures on branch '$current_branch' - check 'gh run list'")
        fi
    fi
}

# Quick status command
show_git_status() {
    print_section "Repository Status"
    
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_warning "Not in a git repository"
        return 0
    fi
    
    # Basic git status
    echo -e "${BLUE}Git Status:${NC}"
    local current_branch=$(git branch --show-current 2>/dev/null || echo "detached")
    local uncommitted=$(git status --porcelain 2>/dev/null | wc -l)
    local unpushed=$(git rev-list --count @{u}..HEAD 2>/dev/null || echo "unknown")
    
    echo -e "  Branch: ${CYAN}$current_branch${NC}"
    echo -e "  Uncommitted changes: ${CYAN}$uncommitted${NC}"
    echo -e "  Unpushed commits: ${CYAN}$unpushed${NC}"
    
    # GitHub status if available
    if command_exists gh; then
        local github_repo=$(gh repo view --json nameWithOwner --jq .nameWithOwner 2>/dev/null || echo "")
        if [ -n "$github_repo" ]; then
            echo -e "\n${BLUE}GitHub Status (${CYAN}$github_repo${BLUE}):${NC}"
            
            # Pull requests
            local open_prs=$(gh pr list --state open --json number,title,author 2>/dev/null)
            local pr_count=$(echo "$open_prs" | jq length 2>/dev/null || echo "0")
            echo -e "  Open pull requests: ${CYAN}$pr_count${NC}"
            
            if [ "$pr_count" -gt 0 ] && [ "$pr_count" -le 5 ]; then
                echo "$open_prs" | jq -r '.[] | "    • #\(.number): \(.title) (@\(.author.login))"' 2>/dev/null | head -3
            fi
            
            # Issues
            local open_issues=$(gh issue list --state open --json number 2>/dev/null | jq length 2>/dev/null || echo "0")
            echo -e "  Open issues: ${CYAN}$open_issues${NC}"
            
            # Recent workflow runs
            local recent_runs=$(gh run list --limit 3 --json status,conclusion,workflowName 2>/dev/null)
            if [ -n "$recent_runs" ] && [ "$recent_runs" != "null" ]; then
                echo -e "  Recent CI/CD runs:"
                echo "$recent_runs" | jq -r '.[] | "    • \(.workflowName): \(.conclusion // .status)"' 2>/dev/null
            fi
        else
            echo -e "\n${YELLOW}  Not authenticated with GitHub CLI or not a GitHub repo${NC}"
        fi
    else
        echo -e "\n${YELLOW}  GitHub CLI (gh) not available for enhanced status${NC}"
    fi
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

# New installer-specific build functions
build_installers() {
    print_section "Building Platform-Specific Installers"
    
    local targets="${1:-all}"
    local enable_signing="${2:-false}"
    local generate_manifests="${3:-false}"
    
    if [ ! -d "desktop" ]; then
        print_error "Desktop directory not found"
        return 1
    fi
    
    if ! command_exists python3; then
        print_error "Python 3 not found - cannot run installer build script"
        return 1
    fi
    
    print_info "Building installers with targets: $targets"
    
    # Prepare build command
    local cmd=("python3" "build_installer.py" "--verbose")
    
    # Add installer targets if specified
    if [ "$targets" != "all" ]; then
        IFS=',' read -ra TARGET_ARRAY <<< "$targets"
        cmd+=("--installer-targets")
        for target in "${TARGET_ARRAY[@]}"; do
            cmd+=("$target")
        done
    fi
    
    # Add code signing if requested
    if [ "$enable_signing" = "true" ]; then
        cmd+=("--code-signing")
    fi
    
    # Add update manifest generation if requested
    if [ "$generate_manifests" = "true" ]; then
        cmd+=("--generate-update-manifest")
    fi
    
    # Run the build
    cd desktop || return 1
    "${cmd[@]}"
    local exit_code=$?
    cd "$PROJECT_ROOT" || return 1
    
    if [ $exit_code -eq 0 ]; then
        print_success "Installers built successfully"
        _show_installer_artifacts
    else
        print_error "Installer build failed"
        return 1
    fi
}

build_msi_installer() {
    print_section "Building Windows MSI Installer"
    build_installers "msi" "${1:-false}" "${2:-false}"
}

build_nsis_installer() {
    print_section "Building Windows NSIS Installer"
    build_installers "nsis" "${1:-false}" "${2:-false}"
}

build_dmg_installer() {
    print_section "Building macOS DMG Installer"
    build_installers "dmg" "${1:-false}" "${2:-false}"
}

build_deb_installer() {
    print_section "Building Linux DEB Package"
    build_installers "deb" "${1:-false}" "${2:-false}"
}

build_rpm_installer() {
    print_section "Building Linux RPM Package"
    build_installers "rpm" "${1:-false}" "${2:-false}"
}

build_appimage_installer() {
    print_section "Building Linux AppImage"
    build_installers "appimage" "${1:-false}" "${2:-false}"
}

build_signed_installers() {
    print_section "Building Signed Installers"
    
    if ! _check_signing_environment; then
        print_error "Code signing environment not properly configured"
        return 1
    fi
    
    build_installers "all" "true" "true"
}

_check_signing_environment() {
    print_info "Checking code signing environment..."
    
    local missing_vars=()
    
    # Check base signing variables
    [ -z "$TAURI_SIGNING_PRIVATE_KEY" ] && missing_vars+=("TAURI_SIGNING_PRIVATE_KEY")
    [ -z "$TAURI_SIGNING_PRIVATE_KEY_PASSWORD" ] && missing_vars+=("TAURI_SIGNING_PRIVATE_KEY_PASSWORD")
    
    # Platform-specific variables
    case "$(uname -s)" in
        "Linux")
            [ -z "$GPG_KEY_ID" ] && missing_vars+=("GPG_KEY_ID")
            ;;
        "Darwin")
            [ -z "$MACOS_SIGNING_IDENTITY" ] && missing_vars+=("MACOS_SIGNING_IDENTITY")
            [ -z "$APPLE_ID" ] && missing_vars+=("APPLE_ID")
            [ -z "$APPLE_PASSWORD" ] && missing_vars+=("APPLE_PASSWORD")
            [ -z "$APPLE_TEAM_ID" ] && missing_vars+=("APPLE_TEAM_ID")
            ;;
        "CYGWIN"*|"MINGW"*|"MSYS"*)
            [ -z "$WINDOWS_CERTIFICATE_PATH" ] && missing_vars+=("WINDOWS_CERTIFICATE_PATH")
            [ -z "$WINDOWS_CERTIFICATE_PASSWORD" ] && missing_vars+=("WINDOWS_CERTIFICATE_PASSWORD")
            ;;
    esac
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        print_error "Missing required environment variables for code signing:"
        for var in "${missing_vars[@]}"; do
            print_error "  - $var"
        done
        print_info "See desktop/frontend/src-tauri/codesign-config.json for requirements"
        return 1
    fi
    
    print_success "Code signing environment is configured"
    return 0
}

_show_installer_artifacts() {
    print_info "Looking for generated installer artifacts..."
    
    local bundle_dir="desktop/frontend/src-tauri/target/release/bundle"
    local found_artifacts=false

    # Helper function to format file sizes
    _format_size() {
        local bytes=$1
        if [ $bytes -lt 1024 ]; then echo "${bytes}B";
        elif [ $bytes -lt 1048576 ]; then echo "$((bytes/1024))K";
        else echo "$((bytes/1048576))M"; fi
    }
    
    if [ -d "$bundle_dir" ]; then
        # Look for various installer types
        local patterns=("*.msi" "*.exe" "*.dmg" "*.deb" "*.rpm" "*.AppImage")
        
        for pattern in "${patterns[@]}"; do
            while IFS= read -r -d '' artifact; do
                if [ -f "$artifact" ]; then
                    local size_bytes=$(wc -c < "$artifact")
                    local size=$(_format_size $size_bytes)

                    print_success "  $(basename "$artifact") ($size)"
                    found_artifacts=true
                fi
            done < <(find "$bundle_dir" -name "$pattern" -print0 2>/dev/null)
        done
    fi
    
    # Check for update manifests
    if [ -d "update-manifests" ]; then
        while IFS= read -r -d '' manifest; do
            if [ -f "$manifest" ]; then
                print_success "  $(basename "$manifest") (update manifest)"
                found_artifacts=true
            fi
        done < <(find "update-manifests" -name "*.json" -print0 2>/dev/null)
    fi
    
    if [ "$found_artifacts" = false ]; then
        print_warning "No installer artifacts found"
    fi
}

generate_update_manifests() {
    print_section "Generating Update Manifests"
    
    if [ ! -f "desktop/generate-update-manifest.py" ]; then
        print_error "Update manifest generator not found"
        return 1
    fi
    
    local version="${1:-1.0.0}"
    local assets_dir="${2:-desktop/frontend/src-tauri/target/release/bundle}"
    
    print_info "Generating update manifests for version $version"
    
    cd desktop || return 1
    python3 generate-update-manifest.py \
        --local-assets "../$assets_dir" \
        --version "$version" \
        --output-dir "../update-manifests" \
        --notes "Release $version"
    local exit_code=$?
    cd "$PROJECT_ROOT" || return 1
    
    if [ $exit_code -eq 0 ]; then
        print_success "Update manifests generated successfully"
        _show_update_manifests
    else
        print_error "Failed to generate update manifests"
        return 1
    fi
}

_show_update_manifests() {
    if [ -d "update-manifests" ]; then
        print_info "Generated update manifests:"
        while IFS= read -r -d '' manifest; do
            if [ -f "$manifest" ]; then
                print_success "  $(basename "$manifest")"
            fi
        done < <(find "update-manifests" -name "*.json" -print0 2>/dev/null)
    fi
}

validate_installer_config() {
    print_section "Validating Installer Configuration"
    
    local config_file="desktop/frontend/src-tauri/tauri.conf.json"
    local codesign_config="desktop/frontend/src-tauri/codesign-config.json"
    local updater_config="desktop/frontend/src-tauri/updater-config.json"
    
    print_info "Checking configuration files..."
    
    # Check main Tauri configuration
    if [ ! -f "$config_file" ]; then
        print_error "Tauri configuration file not found: $config_file"
        return 1
    fi
    
    # Validate JSON syntax
    if command_exists jq; then
        if ! jq empty "$config_file" 2>/dev/null; then
            print_error "Invalid JSON in $config_file"
            return 1
        fi
        print_success "Tauri configuration is valid JSON"
        
        # Check for required bundle configuration
        local bundle_active=$(jq -r '.bundle.active // false' "$config_file")
        if [ "$bundle_active" != "true" ]; then
            print_error "Bundle configuration is not active in $config_file"
            return 1
        fi
        print_success "Bundle configuration is active"
        
        # Check for installer targets
        local targets=$(jq -r '.bundle.targets[]? // empty' "$config_file" 2>/dev/null)
        if [ -z "$targets" ]; then
            print_warning "No installer targets configured in $config_file"
        else
            print_success "Installer targets configured: $(echo "$targets" | tr '\n' ' ')"
        fi
    else
        print_warning "jq not available - skipping JSON validation"
    fi
    
    # Check for installer assets
    local assets_dir="desktop/frontend/src-tauri/installer-assets"
    if [ -d "$assets_dir" ]; then
        print_success "Installer assets directory exists"
    else
        print_warning "Installer assets directory not found: $assets_dir"
        print_info "Placeholder assets will be generated during build"
    fi
    
    # Check code signing configuration
    if [ -f "$codesign_config" ]; then
        print_success "Code signing configuration found"
    else
        print_warning "Code signing configuration not found: $codesign_config"
    fi
    
    # Check updater configuration
    if [ -f "$updater_config" ]; then
        print_success "Auto-updater configuration found"
    else
        print_warning "Auto-updater configuration not found: $updater_config"
    fi
    
    print_success "Installer configuration validation complete"
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
    
    # Root configuration files with Prettier (if available)
    if command_exists npx || command_exists prettier; then
        print_info "Formatting configuration files..."
        if command_exists npx; then
            npx prettier --write "*.md" "*.json" "*.yml" "*.yaml" 2>/dev/null || true
        elif command_exists prettier; then
            prettier --write "*.md" "*.json" "*.yml" "*.yaml" 2>/dev/null || true
        fi
        print_success "Configuration files formatted"
    fi
    
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

# Language-specific test functions
run_python_tests() {
    print_section "Running Python Tests"
    print_info "Running Python tests..."
    run_python_cmd "pytest -v"
    print_success "Python tests completed"
}

run_frontend_tests() {
    print_section "Running Frontend Tests"
    if [ -d "frontend" ]; then
        print_info "Running frontend tests..."
        run_node_cmd "test" "frontend" 2>/dev/null || print_warning "Frontend tests not available or failed"
        print_success "Frontend tests completed"
    else
        print_warning "Frontend directory not found"
    fi
}

run_desktop_tests() {
    print_section "Running Desktop Tests"
    if [ -d "desktop/frontend" ]; then
        print_info "Checking desktop TypeScript..."
        run_node_cmd "check" "desktop/frontend"
        print_success "Desktop tests completed"
    else
        print_warning "Desktop frontend directory not found"
    fi
}

run_rust_tests() {
    print_section "Running Rust Tests"
    if [ -d "desktop/frontend/src-tauri" ] && command_exists cargo; then
        print_info "Running Rust tests..."
        cd desktop/frontend/src-tauri
        cargo test
        cd "$PROJECT_ROOT"
        print_success "Rust tests completed"
    else
        print_warning "Rust tests not available"
    fi
}

# Language-specific linting functions
lint_python() {
    print_section "Linting Python Code"
    print_info "Running flake8..."
    run_python_cmd "flake8 src tests --max-line-length=100" && print_success "Python linting passed" || print_error "Python linting failed"
    print_info "Running mypy..."
    run_python_cmd "mypy src --ignore-missing-imports" && print_success "Python type checking passed" || print_error "Python type checking failed"
}

lint_frontend() {
    print_section "Linting Frontend Code"
    if [ -d "frontend" ]; then
        print_info "Linting frontend..."
        run_node_cmd "lint" "frontend" && print_success "Frontend linting passed" || print_error "Frontend linting failed"
    else
        print_warning "Frontend directory not found"
    fi
}

lint_desktop() {
    print_section "Linting Desktop Code"
    if [ -d "desktop/frontend" ]; then
        print_info "Linting desktop frontend..."
        run_node_cmd "lint" "desktop/frontend" && print_success "Desktop frontend linting passed" || print_error "Desktop frontend linting failed"
    else
        print_warning "Desktop frontend directory not found"
    fi
}

lint_rust() {
    print_section "Linting Rust Code"
    if [ -d "desktop/frontend/src-tauri" ] && command_exists cargo; then
        print_info "Running cargo clippy..."
        cd desktop/frontend/src-tauri
        cargo clippy -- -D warnings && print_success "Rust linting passed" || print_error "Rust linting failed"
        cd "$PROJECT_ROOT"
    else
        print_warning "Rust linting not available"
    fi
}

# Language-specific formatting functions
format_python() {
    print_section "Formatting Python Code"
    print_info "Formatting Python code..."
    run_python_cmd "black src tests"
    run_python_cmd "isort src tests"
    print_success "Python code formatted"
}

format_frontend() {
    print_section "Formatting Frontend Code"
    if [ -d "frontend" ]; then
        print_info "Formatting frontend..."
        run_node_cmd "format" "frontend"
        print_success "Frontend code formatted"
    else
        print_warning "Frontend directory not found"
    fi
}

format_desktop() {
    print_section "Formatting Desktop Code"
    if [ -d "desktop/frontend" ]; then
        print_info "Formatting desktop frontend..."
        run_node_cmd "format" "desktop/frontend"
        print_success "Desktop frontend code formatted"
    else
        print_warning "Desktop frontend directory not found"
    fi
}

format_rust() {
    print_section "Formatting Rust Code"
    if [ -d "desktop/frontend/src-tauri" ] && command_exists cargo; then
        print_info "Formatting Rust code..."
        cd desktop/frontend/src-tauri
        cargo fmt
        cd "$PROJECT_ROOT"
        print_success "Rust code formatted"
    else
        print_warning "Rust formatting not available"
    fi
}

# Language-specific cleaning functions
clean_python() {
    print_section "Cleaning Python Artifacts"
    print_info "Cleaning Python artifacts..."
    rm -rf build dist *.egg-info
    rm -rf .pytest_cache .coverage htmlcov
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete
    print_success "Python artifacts cleaned"
}

clean_frontend() {
    print_section "Cleaning Frontend Artifacts"
    print_info "Cleaning frontend artifacts..."
    [ -d "frontend/node_modules" ] && rm -rf frontend/node_modules
    [ -d "frontend/build" ] && rm -rf frontend/build
    [ -d "frontend/.svelte-kit" ] && rm -rf frontend/.svelte-kit
    [ -d "frontend/dist" ] && rm -rf frontend/dist
    print_success "Frontend artifacts cleaned"
}

clean_desktop() {
    print_section "Cleaning Desktop Artifacts"
    print_info "Cleaning desktop artifacts..."
    [ -d "desktop/frontend/node_modules" ] && rm -rf desktop/frontend/node_modules
    [ -d "desktop/frontend/build" ] && rm -rf desktop/frontend/build
    [ -d "desktop/frontend/.svelte-kit" ] && rm -rf desktop/frontend/.svelte-kit
    [ -d "desktop/frontend/dist" ] && rm -rf desktop/frontend/dist
    print_success "Desktop artifacts cleaned"
}

clean_rust() {
    print_section "Cleaning Rust Artifacts"
    print_info "Cleaning Rust artifacts..."
    [ -d "desktop/frontend/src-tauri/target" ] && rm -rf desktop/frontend/src-tauri/target
    [ -d "desktop/backend/target" ] && rm -rf desktop/backend/target
    print_success "Rust artifacts cleaned"
}

# Cleaning
clean_all() {
    print_section "Cleaning All Build Artifacts"
    clean_python
    clean_frontend
    clean_desktop
    clean_rust
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
    echo -e "  ${GREEN}build [lang]${NC}      Build components (all|python|js|ts|rust)"
    echo -e "  ${GREEN}backend${NC}           Build and validate Python backend only"
    echo -e "  ${GREEN}webapp${NC}            Build SvelteKit web application only"
    echo -e "  ${GREEN}desktop${NC}           Build desktop application (development)"
    echo -e "  ${GREEN}desktop-release${NC}   Build desktop application (release/production)"
    echo ""
    
    echo -e "${YELLOW}Installer Commands:${NC}"
    echo -e "  ${GREEN}installers [type]${NC} Build platform-specific installers"
    echo -e "    Types: all, msi, nsis, dmg, deb, rpm, appimage, signed"
    echo -e "  ${GREEN}sign${NC}              Build signed installers with code signing"
    echo -e "  ${GREEN}update-manifests${NC}  Generate auto-updater manifests"
    echo -e "  ${GREEN}validate-config${NC}   Validate installer configuration"
    echo ""
    
    echo -e "${YELLOW}Development Commands:${NC}"
    echo -e "  ${GREEN}dev-backend${NC}       Start Python MCP server in development mode"
    echo -e "  ${GREEN}dev-webapp${NC}        Start SvelteKit development server"
    echo -e "  ${GREEN}dev-desktop${NC}       Start Tauri desktop application in development"
    echo ""
    
    echo -e "${YELLOW}Quality Assurance:${NC}"
    echo -e "  ${GREEN}test [lang]${NC}       Run tests (all|python|js|ts|rust)"
    echo -e "  ${GREEN}lint [lang]${NC}       Run linting (all|python|js|ts|rust)"
    echo -e "  ${GREEN}format [lang]${NC}     Format code (all|python|js|ts|rust)"
    echo ""
    
    echo -e "${YELLOW}Utility Commands:${NC}"
    echo -e "  ${GREEN}status${NC}            Show detailed git and GitHub repository status"
    echo -e "  ${GREEN}clean [lang]${NC}      Remove build artifacts (all|python|js|ts|rust)"
    echo -e "  ${GREEN}help${NC}              Show this help message"
    echo ""
    
    echo -e "${YELLOW}Examples:${NC}"
    echo -e "  ${CYAN}$0 deps && $0 build${NC}         # Full setup and build"
    echo -e "  ${CYAN}$0 build python${NC}             # Build only Python backend"
    echo -e "  ${CYAN}$0 test ts${NC}                  # Test only TypeScript/desktop"
    echo -e "  ${CYAN}$0 lint rust${NC}                # Lint only Rust code"
    echo -e "  ${CYAN}$0 clean js${NC}                 # Clean only JavaScript artifacts"
    echo -e "  ${CYAN}$0 dev-desktop${NC}              # Start desktop development"
    echo -e "  ${CYAN}$0 format python && $0 test py${NC} # Format and test Python only"
    echo ""
    
    echo -e "${YELLOW}Installer Examples:${NC}"
    echo -e "  ${CYAN}$0 installers${NC}               # Build all platform installers"
    echo -e "  ${CYAN}$0 installers msi${NC}           # Build Windows MSI installer only"
    echo -e "  ${CYAN}$0 installers dmg true${NC}      # Build macOS DMG with signing"
    echo -e "  ${CYAN}$0 sign${NC}                     # Build signed installers + manifests"
    echo -e "  ${CYAN}$0 validate-config${NC}          # Check installer configuration"
    echo -e "  ${CYAN}$0 update-manifests 1.2.0${NC}   # Generate update manifests for v1.2.0"
    echo ""
    
    echo -e "${YELLOW}Detected Tools:${NC}"
    echo -e "  Python Manager: ${CYAN}$(detect_python_manager)${NC}"
    echo -e "  Node.js Manager: ${CYAN}$(detect_node_manager)${NC}"
    echo -e "  Rust/Cargo: ${CYAN}$(command_exists cargo && echo "available" || echo "not found")${NC}"
    echo -e "  GitHub CLI: ${CYAN}$(command_exists gh && echo "available (enhanced git status)" || echo "not found")${NC}"
    echo -e "  JSON Parser: ${CYAN}$(command_exists jq && echo "available" || echo "not found (limited GitHub features)")${NC}"
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
        case "${2:-all}" in
            "python"|"backend"|"py")
                build_backend
                ;;
            "javascript"|"js"|"frontend"|"webapp"|"web")
                build_webapp
                ;;
            "typescript"|"ts"|"desktop")
                build_desktop
                ;;
            "rust"|"tauri")
                build_desktop
                ;;
            "all"|*)
                build_backend
                build_webapp
                build_desktop
                print_success "All components built successfully!"
                ;;
        esac
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
    
    "installers")
        print_header
        case "${2:-all}" in
            "msi")
                build_msi_installer "${3:-false}" "${4:-false}"
                ;;
            "nsis")
                build_nsis_installer "${3:-false}" "${4:-false}"
                ;;
            "dmg")
                build_dmg_installer "${3:-false}" "${4:-false}"
                ;;
            "deb")
                build_deb_installer "${3:-false}" "${4:-false}"
                ;;
            "rpm")
                build_rpm_installer "${3:-false}" "${4:-false}"
                ;;
            "appimage")
                build_appimage_installer "${3:-false}" "${4:-false}"
                ;;
            "signed")
                build_signed_installers
                ;;
            "all"|*)
                build_installers "all" "${3:-false}" "${4:-false}"
                ;;
        esac
        ;;
    
    "update-manifests")
        print_header
        generate_update_manifests "${2:-1.0.0}" "${3:-}"
        ;;
    
    "validate-config")
        print_header
        validate_installer_config
        ;;
    
    "sign")
        print_header
        build_signed_installers
        ;;
    
    "test")
        print_header
        case "${2:-all}" in
            "python"|"backend"|"py")
                run_python_tests
                ;;
            "javascript"|"js"|"frontend"|"webapp"|"web")
                run_frontend_tests
                ;;
            "typescript"|"ts"|"desktop")
                run_desktop_tests
                ;;
            "rust"|"tauri")
                run_rust_tests
                ;;
            "all"|*)
                run_tests
                ;;
        esac
        ;;
    
    "lint")
        print_header
        case "${2:-all}" in
            "python"|"backend"|"py")
                lint_python
                ;;
            "javascript"|"js"|"frontend"|"webapp"|"web")
                lint_frontend
                ;;
            "typescript"|"ts"|"desktop")
                lint_desktop
                ;;
            "rust"|"tauri")
                lint_rust
                ;;
            "all"|*)
                lint_all
                ;;
        esac
        ;;
    
    "format")
        print_header
        case "${2:-all}" in
            "python"|"backend"|"py")
                format_python
                ;;
            "javascript"|"js"|"frontend"|"webapp"|"web")
                format_frontend
                ;;
            "typescript"|"ts"|"desktop")
                format_desktop
                ;;
            "rust"|"tauri")
                format_rust
                ;;
            "all"|*)
                format_all
                ;;
        esac
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
    
    "status")
        show_git_status
        ;;
    
    "clean")
        print_header
        case "${2:-all}" in
            "python"|"backend"|"py")
                clean_python
                ;;
            "javascript"|"js"|"frontend"|"webapp"|"web")
                clean_frontend
                ;;
            "typescript"|"ts"|"desktop")
                clean_desktop
                ;;
            "rust"|"tauri")
                clean_rust
                ;;
            "all"|*)
                clean_all
                ;;
        esac
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