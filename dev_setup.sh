#!/bin/bash
# MDMAI Development Environment Setup Script
# This script fixes common development setup issues

set -e  # Exit on any error

echo "🚀 Setting up MDMAI development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found. Please run this script from the MDMAI project root."
    exit 1
fi

print_info "Current directory: $(pwd)"

# 1. Clean up any existing virtual environments
print_info "Cleaning up existing virtual environments..."
rm -rf venv test_venv .venv
print_status "Cleaned up old virtual environments"

# 2. Create new virtual environment
print_info "Creating new Python virtual environment..."
PYTHON_CMD=$(command -v python3 || command -v python)
if [ -z "$PYTHON_CMD" ]; then
    print_error "Python not found. Please install Python 3."
    exit 1
fi
"$PYTHON_CMD" -m venv venv
print_status "Created virtual environment: venv/"

# 3. Activate and upgrade pip
print_info "Activating virtual environment and upgrading pip..."
source venv/bin/activate
if [ -z "$VIRTUAL_ENV" ]; then
    print_error "Failed to activate virtual environment"
    exit 1
fi
pip install --upgrade pip setuptools wheel
print_status "Upgraded pip and setuptools"

# 4. Install core dependencies (subset for faster setup)
print_info "Installing core Python dependencies..."
pip install fastmcp==2.11.3 structlog==24.1.0 fastapi==0.109.0 pytest==7.4.0 mypy==1.7.0 ruff==0.1.0
print_status "Installed core dependencies"

# 5. Test FastMCP import
print_info "Testing FastMCP import..."
python -c "from fastmcp import FastMCP; print(f'✅ FastMCP {FastMCP.__version__ if hasattr(FastMCP, \"__version__\") else \"imported\"} works correctly')" || {
    print_warning "FastMCP import test failed, trying alternative import..."
    python -c "from mcp.server.fastmcp import FastMCP; print('✅ FastMCP imported successfully (legacy path)')"
}

# 6. Test structlog import
print_info "Testing structlog import..."
python -c "import structlog; print('✅ structlog imported successfully')"

# 7. Setup frontend if node is available
if command -v npm &> /dev/null; then
    print_info "Setting up frontend dependencies..."
    cd frontend
    npm install
    print_status "Frontend dependencies installed"
    cd ..
else
    print_warning "npm not found, skipping frontend setup"
fi

# 8. Create convenience scripts
print_info "Creating convenience scripts..."

cat > run_tests.sh << 'EOF'
#!/bin/bash
# Run tests with proper environment
source venv/bin/activate
python -m pytest tests/ -v
EOF
chmod +x run_tests.sh

cat > run_linting.sh << 'EOF'
#!/bin/bash
# Run linting checks
source venv/bin/activate
ruff check src/ --select=E,F --statistics
EOF
chmod +x run_linting.sh

cat > run_typecheck.sh << 'EOF'
#!/bin/bash
# Run type checking
source venv/bin/activate
mypy src/main.py --ignore-missing-imports
EOF
chmod +x run_typecheck.sh

print_status "Created convenience scripts: run_tests.sh, run_linting.sh, run_typecheck.sh"

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To install full dependencies (may take longer):"
echo "  source venv/bin/activate && pip install -r requirements.txt"
echo ""
echo "To run tests:"
echo "  ./run_tests.sh"
echo ""
echo "To check code quality:"
echo "  ./run_linting.sh"
echo ""