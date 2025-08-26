.PHONY: help install install-dev test test-fast test-cov lint format clean run setup-uv setup-poetry

# Default target
help:
	@echo "TTRPG Assistant MCP Server - Development Commands"
	@echo "================================================="
	@echo ""
	@echo "Setup Commands:"
	@echo "  make setup-uv      - Install uv package manager (recommended)"
	@echo "  make setup-poetry  - Install poetry package manager (alternative)"
	@echo "  make install       - Install production dependencies"
	@echo "  make install-dev   - Install all dependencies (CPU-only, ~200MB)"
	@echo ""
	@echo "GPU Support (choose one):"
	@echo "  make install-cuda  - Install NVIDIA GPU support (~2GB)"
	@echo "  make install-rocm  - Install AMD GPU support"
	@echo "  make install-cpu-torch - Install CPU-only PyTorch (default)"
	@echo ""
	@echo "Development Commands:"
	@echo "  make test         - Run all tests with coverage"
	@echo "  make test-fast    - Run tests without coverage (faster)"
	@echo "  make test-unit    - Run only unit tests"
	@echo "  make test-int     - Run only integration tests"
	@echo "  make lint         - Run linting checks (flake8, mypy, pylint)"
	@echo "  make format       - Format code with black and isort"
	@echo "  make clean        - Remove build artifacts and cache"
	@echo ""
	@echo "Running Commands:"
	@echo "  make run          - Run the MCP server"
	@echo "  make run-dev      - Run in development mode with auto-reload"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run in Docker container"

# Setup uv (recommended - fast and modern)
setup-uv:
	@echo "Installing uv package manager..."
	@curl -LsSf https://astral.sh/uv/install.sh | sh
	@echo "✅ uv installed successfully!"
	@echo "Run 'make install-dev' to install dependencies"

# Setup poetry (alternative)
setup-poetry:
	@echo "Installing poetry package manager..."
	@curl -sSL https://install.python-poetry.org | python3 -
	@echo "✅ Poetry installed successfully!"
	@echo "Run 'poetry install' to install dependencies"

# Install production dependencies with uv
install:
	@if command -v uv >/dev/null 2>&1; then \
		echo "Installing production dependencies with uv..."; \
		if [ ! -d ".venv" ]; then \
			uv venv; \
		fi; \
		uv pip install -e . 2>&1 | grep -v "warning: A virtual environment already exists" || true; \
	elif command -v poetry >/dev/null 2>&1; then \
		echo "Installing production dependencies with poetry..."; \
		poetry install --only main; \
	else \
		echo "Installing with pip (in virtual environment)..."; \
		if [ ! -d ".venv" ]; then \
			python -m venv .venv; \
		fi; \
		. .venv/bin/activate && \
		pip install --upgrade pip --quiet && \
		pip install -e . --quiet --upgrade; \
	fi

# Install all dependencies including dev tools (CPU-only by default)
install-dev: install-cpu-torch
	@if command -v uv >/dev/null 2>&1; then \
		echo "Installing all dependencies with uv (CPU-only)..."; \
		if [ ! -d ".venv" ]; then \
			uv venv; \
		fi; \
		uv pip install -e ".[dev,test,docs]" 2>&1 | grep -v "warning: A virtual environment already exists" || true; \
		echo "✅ Dependencies installed! Activate with: source .venv/bin/activate"; \
	elif command -v poetry >/dev/null 2>&1; then \
		echo "Installing all dependencies with poetry..."; \
		poetry install --with dev,test,docs; \
		echo "✅ Dependencies installed! Activate with: poetry shell"; \
	else \
		echo "Installing with pip (in virtual environment)..."; \
		if [ ! -d ".venv" ]; then \
			python -m venv .venv; \
		fi; \
		. .venv/bin/activate && \
		pip install --upgrade pip --quiet && \
		pip install -e ".[dev,test,docs]" --quiet --upgrade; \
		echo "✅ Dependencies installed! Activate with: source .venv/bin/activate"; \
	fi

# Install CPU-only PyTorch (small, no GPU support)
install-cpu-torch:
	@echo "Installing CPU-only PyTorch (no GPU support, ~200MB)..."
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && \
		pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet; \
	elif command -v uv >/dev/null 2>&1; then \
		if [ ! -d ".venv" ]; then \
			uv venv; \
		fi; \
		uv pip install torch --index-url https://download.pytorch.org/whl/cpu; \
	else \
		pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet; \
	fi
	@echo "✅ CPU-only PyTorch installed"

# Install CUDA PyTorch (NVIDIA GPU support, ~2GB)
install-cuda:
	@echo "Installing CUDA-enabled PyTorch (NVIDIA GPU support, ~2GB)..."
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && \
		pip install torch --quiet; \
	elif command -v uv >/dev/null 2>&1; then \
		uv pip install torch; \
	else \
		pip install torch --quiet; \
	fi
	@echo "✅ CUDA PyTorch installed"

# Install ROCm PyTorch (AMD GPU support)
install-rocm:
	@echo "Installing ROCm-enabled PyTorch (AMD GPU support)..."
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && \
		pip install torch --index-url https://download.pytorch.org/whl/rocm6.2 --quiet; \
	elif command -v uv >/dev/null 2>&1; then \
		uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.2; \
	else \
		pip install torch --index-url https://download.pytorch.org/whl/rocm6.2 --quiet; \
	fi
	@echo "✅ ROCm PyTorch installed"

# Run tests with coverage
test:
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && pytest; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry run pytest; \
	elif command -v uv >/dev/null 2>&1; then \
		uv run pytest; \
	else \
		pytest; \
	fi

# Run tests without coverage (faster)
test-fast:
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && pytest -x --tb=short --no-cov; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry run pytest -x --tb=short --no-cov; \
	elif command -v uv >/dev/null 2>&1; then \
		uv run pytest -x --tb=short --no-cov; \
	else \
		pytest -x --tb=short --no-cov; \
	fi

# Run only unit tests
test-unit:
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && pytest -m "unit" -v; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry run pytest -m "unit" -v; \
	elif command -v uv >/dev/null 2>&1; then \
		uv run pytest -m "unit" -v; \
	else \
		pytest -m "unit" -v; \
	fi

# Run only integration tests
test-int:
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && pytest -m "integration" -v; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry run pytest -m "integration" -v; \
	elif command -v uv >/dev/null 2>&1; then \
		uv run pytest -m "integration" -v; \
	else \
		pytest -m "integration" -v; \
	fi

# Run parallel tests
test-parallel:
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && pytest -n auto; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry run pytest -n auto; \
	elif command -v uv >/dev/null 2>&1; then \
		uv run pytest -n auto; \
	else \
		pytest -n auto; \
	fi

# View test coverage report
test-cov:
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && pytest --cov-report=html && open htmlcov/index.html; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry run pytest --cov-report=html && open htmlcov/index.html; \
	elif command -v uv >/dev/null 2>&1; then \
		uv run pytest --cov-report=html && open htmlcov/index.html; \
	else \
		pytest --cov-report=html && open htmlcov/index.html; \
	fi

# Linting
lint:
	@echo "Running flake8..."
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && flake8 src tests; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry run flake8 src tests; \
	elif command -v uv >/dev/null 2>&1; then \
		uv run flake8 src tests; \
	else \
		flake8 src tests; \
	fi
	@echo "Running mypy..."
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && mypy src; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry run mypy src; \
	elif command -v uv >/dev/null 2>&1; then \
		uv run mypy src; \
	else \
		mypy src; \
	fi

# Format code
format:
	@echo "Formatting with black..."
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && black src tests; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry run black src tests; \
	elif command -v uv >/dev/null 2>&1; then \
		uv run black src tests; \
	else \
		black src tests; \
	fi
	@echo "Sorting imports with isort..."
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && isort src tests; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry run isort src tests; \
	elif command -v uv >/dev/null 2>&1; then \
		uv run isort src tests; \
	else \
		isort src tests; \
	fi

# Check formatting without changing files
check-format:
	@echo "Checking black formatting..."
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && black --check src tests; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry run black --check src tests; \
	elif command -v uv >/dev/null 2>&1; then \
		uv run black --check src tests; \
	else \
		black --check src tests; \
	fi
	@echo "Checking import sorting..."
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && isort --check-only src tests; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry run isort --check-only src tests; \
	elif command -v uv >/dev/null 2>&1; then \
		uv run isort --check-only src tests; \
	else \
		isort --check-only src tests; \
	fi

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf build dist *.egg-info
	@rm -rf .pytest_cache .coverage htmlcov
	@rm -rf **/__pycache__ **/*.pyc
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete
	@echo "✅ Cleaned!"

# Run the server
run:
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && python -m src.main; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry run python -m src.main; \
	elif command -v uv >/dev/null 2>&1; then \
		uv run python -m src.main; \
	else \
		python -m src.main; \
	fi

# Run in development mode
run-dev:
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && MCP_STDIO_MODE=false python -m src.main; \
	elif command -v poetry >/dev/null 2>&1; then \
		MCP_STDIO_MODE=false poetry run python -m src.main; \
	elif command -v uv >/dev/null 2>&1; then \
		MCP_STDIO_MODE=false uv run python -m src.main; \
	else \
		MCP_STDIO_MODE=false python -m src.main; \
	fi

# Build Docker image
docker-build:
	@docker build -t ttrpg-assistant:latest .

# Run in Docker
docker-run:
	@docker run -it --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/logs:/app/logs \
		-p 8000:8000 \
		ttrpg-assistant:latest

# Pre-commit hooks
pre-commit:
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && pre-commit run --all-files; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry run pre-commit run --all-files; \
	elif command -v uv >/dev/null 2>&1; then \
		uv run pre-commit run --all-files; \
	else \
		pre-commit run --all-files; \
	fi

# Install pre-commit hooks
pre-commit-install:
	@if [ -f .venv/bin/activate ]; then \
		. .venv/bin/activate && pre-commit install; \
	elif command -v poetry >/dev/null 2>&1; then \
		poetry run pre-commit install; \
	elif command -v uv >/dev/null 2>&1; then \
		uv run pre-commit install; \
	else \
		pre-commit install; \
	fi

# Deployment Commands
deploy-install:
	@echo "Installing TTRPG Assistant MCP Server..."
	@sudo bash deploy/scripts/install.sh

deploy-configure:
	@echo "Running configuration wizard..."
	@python deploy/scripts/configure.py

deploy-check:
	@echo "Checking deployment prerequisites..."
	@python deploy/scripts/setup_environment.py --json-output

deploy-migrate:
	@echo "Running database migration..."
	@python deploy/migration/migrate.py $(VERSION)

deploy-backup:
	@echo "Creating backup..."
	@python deploy/backup/backup_manager.py --create

deploy-restore:
	@echo "Restoring from backup..."
	@python deploy/backup/restore_manager.py --restore $(BACKUP_ID)

deploy-package:
	@echo "Creating deployment package..."
	@rm -rf dist/
	@mkdir -p dist/
	@python setup.py sdist bdist_wheel
	@tar czf dist/ttrpg-assistant-$(shell python -c "import setup; print(setup.version)").tar.gz \
		--exclude='.git' --exclude='venv' --exclude='__pycache__' \
		--exclude='*.pyc' --exclude='.pytest_cache' \
		--exclude='htmlcov' --exclude='dist' \
		.
	@echo "✓ Deployment package created in dist/"

deploy-docker-build:
	@echo "Building Docker image with deployment tools..."
	@docker build -f Dockerfile -t ttrpg-assistant:latest \
		--build-arg GPU_SUPPORT=$(GPU_SUPPORT) .
	@echo "✓ Docker image built: ttrpg-assistant:latest"

deploy-docker-push:
	@echo "Pushing Docker image to registry..."
	@docker tag ttrpg-assistant:latest $(REGISTRY)/ttrpg-assistant:latest
	@docker push $(REGISTRY)/ttrpg-assistant:latest
	@echo "✓ Docker image pushed to $(REGISTRY)"

deploy-systemd:
	@echo "Installing systemd service..."
	@sudo cp deploy/config/systemd.service.template /etc/systemd/system/ttrpg-assistant.service
	@sudo systemctl daemon-reload
	@sudo systemctl enable ttrpg-assistant.service
	@echo "✓ Systemd service installed"

deploy-status:
	@echo "Checking deployment status..."
	@systemctl status ttrpg-assistant || true
	@echo ""
	@echo "Version Information:"
	@python -c "from deploy.migration.version_manager import VersionManager; \
		import json; \
		vm = VersionManager('.'); \
		print(json.dumps(vm.get_version_info(), indent=2))"

deploy-health-check:
	@echo "Running health checks..."
	@curl -f http://localhost:8000/health || echo "Health check failed"

deploy-logs:
	@echo "Viewing deployment logs..."
	@if [ -f /var/log/ttrpg-assistant/ttrpg-assistant.log ]; then \
		tail -f /var/log/ttrpg-assistant/ttrpg-assistant.log; \
	elif [ -f logs/ttrpg-assistant.log ]; then \
		tail -f logs/ttrpg-assistant.log; \
	else \
		echo "No log file found"; \
	fi

deploy-uninstall:
	@echo "Uninstalling TTRPG Assistant MCP Server..."
	@if [ -f /opt/ttrpg-assistant/uninstall.sh ]; then \
		sudo /opt/ttrpg-assistant/uninstall.sh; \
	else \
		echo "Uninstall script not found"; \
	fi