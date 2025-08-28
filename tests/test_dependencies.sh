#!/bin/bash
# Test runner script for dependency updates
#
# Usage:
#   ./test_dependencies.sh [options]
#
# Options:
#   --full     Run full test suite including stress tests
#   --quick    Run quick tests only
#   --coverage Generate coverage report
#   --help     Show this help message

set -e  # Exit on error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
QUICK_MODE=false
COVERAGE=false
VERBOSE=false
PARALLEL=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            QUICK_MODE=false
            COVERAGE=true
            shift
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --help|-h)
            echo "Dependency Update Test Runner"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --full     Run full test suite including stress tests"
            echo "  --quick    Run quick tests only"
            echo "  --coverage Generate coverage report"
            echo "  --verbose  Verbose output"
            echo "  --parallel Run tests in parallel"
            echo "  --help     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage information"
            exit 1
            ;;
    esac
done

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Header
print_status "$BLUE" "=========================================="
print_status "$BLUE" "     Dependency Update Test Suite        "
print_status "$BLUE" "=========================================="
echo ""

# Check Python version
print_status "$YELLOW" "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    print_status "$GREEN" "✓ Python $python_version (>= $required_version required)"
else
    print_status "$RED" "✗ Python $python_version is too old (>= $required_version required)"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    print_status "$YELLOW" "⚠️  No virtual environment detected"
    echo "   Consider activating a virtual environment first:"
    echo "   source venv/bin/activate"
    echo ""
fi

# Build test command
TEST_CMD="python3 tests/run_dependency_tests.py"

if [ "$QUICK_MODE" = true ]; then
    TEST_CMD="$TEST_CMD --quick"
fi

if [ "$COVERAGE" = true ]; then
    TEST_CMD="$TEST_CMD --coverage"
fi

if [ "$VERBOSE" = true ]; then
    TEST_CMD="$TEST_CMD --verbose"
fi

if [ "$PARALLEL" = true ]; then
    TEST_CMD="$TEST_CMD --parallel"
fi

TEST_CMD="$TEST_CMD --report"

# Run the tests
print_status "$YELLOW" "Running dependency tests..."
echo "Command: $TEST_CMD"
echo ""

# Change to project root directory
cd "$(dirname "$0")/.."

# Run tests and capture exit code
if $TEST_CMD; then
    EXIT_CODE=0
    print_status "$GREEN" ""
    print_status "$GREEN" "=========================================="
    print_status "$GREEN" "        All Tests Completed Successfully  "
    print_status "$GREEN" "=========================================="
else
    EXIT_CODE=$?
    print_status "$RED" ""
    print_status "$RED" "=========================================="
    print_status "$RED" "           Some Tests Failed              "
    print_status "$RED" "=========================================="
fi

# Show report locations
echo ""
print_status "$BLUE" "Reports generated:"
echo "  • Test report: dependency_test_report.json"
echo "  • HTML report: test_report_deps.html"

if [ "$COVERAGE" = true ]; then
    echo "  • Coverage report: htmlcov_deps/index.html"
fi

exit $EXIT_CODE