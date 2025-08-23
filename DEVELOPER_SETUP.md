# Developer Setup Guide

## Quick Start

Run the automated setup script:
```bash
./quick_setup.sh
```

This will:
1. Detect your preferred package manager (uv, Poetry, or pip)
2. Detect your GPU hardware (NVIDIA, AMD, or CPU-only)
3. Install appropriate dependencies with optimized PyTorch version
4. Set up your development environment automatically

### GPU Support

The setup script automatically detects your hardware and installs the appropriate PyTorch version:

- **CPU-only** (default): ~200MB download, no GPU acceleration
- **NVIDIA GPU**: ~2GB download with CUDA support (auto-detected)
- **AMD GPU**: ROCm support (requires ROCm installation)

To manually change GPU support after installation:
```bash
make install-cpu-torch  # Switch to CPU-only (smaller)
make install-cuda       # Switch to NVIDIA GPU support
make install-rocm       # Switch to AMD GPU support
```

## Manual Setup Options

### Option 1: Using uv (Recommended - Fastest)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
uv pip install -e ".[dev,test,docs]"

# Activate environment
source .venv/bin/activate
```

### Option 2: Using Poetry

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install --with dev,test,docs

# Activate environment
poetry shell
```

### Option 3: Using pip with venv

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -e ".[dev,test,docs]"
```

## Available Make Commands

After setting up your environment, you can use these commands:

### Setup Commands
- `make setup-uv` - Install uv package manager
- `make setup-poetry` - Install Poetry package manager
- `make install` - Install production dependencies only
- `make install-dev` - Install all dependencies including dev tools

### Development Commands
- `make test` - Run all tests with coverage
- `make test-fast` - Run tests without coverage (faster)
- `make test-unit` - Run only unit tests
- `make test-int` - Run only integration tests
- `make test-parallel` - Run tests in parallel
- `make lint` - Run linting checks (flake8, mypy)
- `make format` - Format code with black and isort
- `make check-format` - Check formatting without changes
- `make clean` - Remove build artifacts and cache

### Running Commands
- `make run` - Run the MCP server
- `make run-dev` - Run in development mode with auto-reload

### Docker Commands
- `make docker-build` - Build Docker image
- `make docker-run` - Run in Docker container

### Git Hooks
- `make pre-commit` - Run pre-commit hooks on all files
- `make pre-commit-install` - Install pre-commit hooks

## Development Workflow

1. **Setup Environment**
   ```bash
   ./quick_setup.sh
   # or
   make install-dev
   ```

2. **Before Committing**
   ```bash
   make format      # Format code
   make lint        # Check code quality
   make test        # Run tests
   ```

3. **Running the Server**
   ```bash
   make run         # Production mode
   make run-dev     # Development mode
   ```

## Dependency Management

### Adding Dependencies

**Production dependency:**
```bash
# Add to pyproject.toml dependencies list
# Then reinstall
make install
```

**Development dependency:**
```bash
# Add to pyproject.toml dev/test/docs groups
# Then reinstall
make install-dev
```

### Updating Dependencies

With uv:
```bash
uv pip install --upgrade package-name
```

With Poetry:
```bash
poetry update package-name
```

## Testing

### Run All Tests
```bash
make test
```

### Run Specific Test Categories
```bash
make test-unit       # Unit tests only
make test-int        # Integration tests only
make test-parallel   # Run tests in parallel
```

### Generate Coverage Report
```bash
make test-cov        # Generates HTML coverage report
```

## Code Quality

### Formatting
```bash
make format          # Auto-format code
make check-format    # Check formatting without changes
```

### Linting
```bash
make lint           # Run all linters (flake8, mypy, pylint)
```

### Pre-commit Hooks
```bash
make pre-commit-install  # Install hooks
make pre-commit          # Run hooks manually
```

## Common Issues

### "This is a managed environment" error
This project uses proper virtual environment management. Always activate your environment before running commands:
- uv/pip: `source .venv/bin/activate`
- Poetry: `poetry shell`

### Missing dependencies
```bash
make install-dev    # Reinstall all dependencies
```

### Tests failing due to missing imports
Some tests may require optional dependencies. Install all test dependencies:
```bash
make install-dev
```

### Slow dependency installation
Use uv for faster installation:
```bash
make setup-uv
make install-dev
```

## IDE Setup

### VS Code
1. Install Python extension
2. Select interpreter: `.venv/bin/python`
3. Enable formatting on save with Black
4. Enable linting with flake8 and mypy

### PyCharm
1. Set Project Interpreter to `.venv/bin/python`
2. Enable Black formatter
3. Configure pytest as test runner

## Contributing

1. Create a feature branch
2. Make your changes
3. Run `make format lint test`
4. Commit with descriptive message
5. Push and create pull request

## Support

For issues or questions:
- Check [GitHub Issues](https://github.com/Raudbjorn/MDMAI/issues)
- Review [Project Wiki](https://github.com/Raudbjorn/MDMAI/wiki)
- Contact the development team