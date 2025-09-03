# Test Suite Organization

This directory contains a well-organized test suite for the TTRPG Assistant project, structured for clarity, maintainability, and efficient execution.

## Directory Structure

```
tests/
├── unit/                           # Unit tests (fast, isolated)
│   ├── ai_providers/              # AI provider tests
│   ├── bridge/                    # MCP bridge tests  
│   ├── campaign/                  # Campaign management tests
│   ├── mcp/                       # MCP tools tests
│   ├── pdf_processing/            # PDF processing tests
│   ├── personality/               # Personality system tests
│   ├── search/                    # Search engine tests
│   ├── security/                  # Security component tests
│   └── utils/                     # Utility function tests
├── integration/                   # Integration tests (moderate speed)
│   ├── test_database_integration.py
│   ├── test_ebook_integration.py
│   └── ...
├── e2e/                          # End-to-end tests (slower)
│   └── test_web_interface_e2e.py
├── load/                         # Performance and load tests
│   ├── test_load_performance.py
│   └── test_stress.py
├── regression/                   # Regression tests
│   ├── test_code_review_fixes.py
│   └── ...
├── bridge/                       # Bridge service tests
├── security/                     # Security-focused tests
├── conftest.py                   # Shared fixtures and configuration
├── test_runner.py               # Comprehensive test runner
└── README.md                    # This file
```

## Test Categories

### Unit Tests (`tests/unit/`)
- **Fast execution** (< 1 second per test)
- **Isolated components** - no external dependencies
- **High coverage** - aim for 80%+ coverage of core functionality
- **Mocked dependencies** - use mocks for external services

### Integration Tests (`tests/integration/`)
- **Component interaction** - test how components work together
- **Real dependencies** - may use real databases/services in test mode
- **Moderate speed** (1-10 seconds per test)
- **End-to-end workflows** within specific domains

### End-to-End Tests (`tests/e2e/`)
- **Full system tests** - complete user workflows
- **Real environment** - closest to production setup
- **Slower execution** (10+ seconds per test)
- **UI/API integration** - test complete request/response cycles

### Load Tests (`tests/load/`)
- **Performance testing** - measure response times and throughput
- **Stress testing** - find system limits
- **Memory testing** - detect memory leaks
- **Concurrent testing** - test under high load

### Regression Tests (`tests/regression/`)
- **Bug fix verification** - ensure fixed bugs stay fixed
- **Compatibility testing** - test against different versions
- **Edge case testing** - unusual scenarios that previously failed

### Security Tests (`tests/security/`)
- **Authentication/authorization** testing
- **Input validation** and sanitization
- **Encryption/decryption** verification
- **Access control** testing

## Running Tests

### Using the Test Runner (Recommended)

```bash
# Run all tests
python tests/test_runner.py

# Run specific category
python tests/test_runner.py unit
python tests/test_runner.py integration
python tests/test_runner.py security

# Run with options
python tests/test_runner.py unit --fast --verbose
python tests/test_runner.py integration --no-coverage
python tests/test_runner.py load --parallel
```

### Using build.sh (Project Standard)

```bash
# Run all tests
./build.sh test

# Run Python tests only
./build.sh test python

# Run with different languages
./build.sh test js
./build.sh test rust
```

### Using pytest directly

```bash
# Run all tests
pytest

# Run specific directory
pytest tests/unit/security/

# Run with markers
pytest -m "security"
pytest -m "not slow"

# Run specific test
pytest tests/unit/security/test_secure_credential_management.py::TestCredentialEncryption::test_encryption_decryption_roundtrip
```

## Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests  
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.load` - Load/performance tests
- `@pytest.mark.security` - Security tests
- `@pytest.mark.slow` - Slow tests (> 5 seconds)
- `@pytest.mark.asyncio` - Async tests
- `@pytest.mark.requires_redis` - Requires Redis
- `@pytest.mark.requires_docker` - Requires Docker

## Writing Tests

### Test Organization Principles

1. **Mirror source structure** - `tests/unit/security/` mirrors `src/security/`
2. **One test file per module** - `test_credential_manager.py` tests `credential_manager.py`
3. **Group related tests** - Use test classes to group related functionality
4. **Clear naming** - Test names should describe what they test

### Test Naming Conventions

```python
# Test files
test_module_name.py

# Test classes (optional, for grouping)
class TestModuleName:
    class TestSpecificFunction:

# Test functions
def test_function_behavior():
def test_function_with_invalid_input():
def test_function_error_handling():
```

### Example Test Structure

```python
"""Tests for credential_manager module."""

import pytest
from unittest.mock import Mock, AsyncMock
from src.security.credential_manager import SecureCredentialManager

class TestSecureCredentialManager:
    """Tests for the SecureCredentialManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create a test manager instance."""
        config = Mock()
        return SecureCredentialManager(config)
    
    @pytest.mark.asyncio
    async def test_store_credential_success(self, manager):
        """Test successful credential storage."""
        # Test implementation
        pass
    
    @pytest.mark.asyncio
    async def test_store_credential_invalid_input(self, manager):
        """Test credential storage with invalid input."""
        # Test implementation
        pass
        
    @pytest.mark.security
    @pytest.mark.slow
    async def test_encryption_security_properties(self, manager):
        """Test security properties of encryption."""
        # Test implementation
        pass
```

## Configuration

### pytest.ini
Main pytest configuration with:
- Test discovery patterns
- Coverage settings
- Markers definition
- Command line defaults

### conftest.py
Shared fixtures including:
- Database fixtures
- Mock objects
- Test clients
- Cleanup utilities

## Coverage Reports

Test coverage reports are generated in multiple formats:
- **Terminal**: Summary during test run
- **HTML**: `htmlcov/index.html` - detailed interactive report
- **XML**: `coverage.xml` - for CI/CD integration

## Continuous Integration

The test suite is designed for CI/CD with:
- **Fast feedback** - unit tests run first
- **Parallel execution** - tests can run in parallel
- **Selective execution** - only run affected tests
- **Clear reporting** - structured output for CI systems

## Best Practices

1. **Test Pyramid** - More unit tests, fewer integration/e2e tests
2. **Independent Tests** - Tests should not depend on each other
3. **Deterministic** - Tests should always pass/fail consistently  
4. **Fast Feedback** - Unit tests should run quickly
5. **Clear Failures** - Test failures should clearly indicate the problem
6. **Isolated** - Tests should not affect each other or global state
7. **Realistic** - Integration tests should use realistic test data
8. **Comprehensive** - Cover happy path, edge cases, and error conditions

## Debugging Tests

```bash
# Run with verbose output
pytest -v

# Run with print statements
pytest -s

# Run specific test with debugging
pytest tests/unit/security/test_credential_encryption.py::test_specific_function -v -s

# Run with debugger
pytest --pdb

# Run last failed tests
pytest --lf
```

## Performance

- **Unit tests**: Should complete in < 30 seconds total
- **Integration tests**: Should complete in < 5 minutes total  
- **Full test suite**: Should complete in < 15 minutes total
- **Parallel execution**: Use `-n auto` for faster execution

## Contributing

When adding new tests:

1. **Choose the right category** - unit vs integration vs e2e
2. **Add appropriate markers** - security, slow, etc.
3. **Follow naming conventions** - clear, descriptive names
4. **Include documentation** - docstrings explaining test purpose
5. **Test edge cases** - not just happy path
6. **Keep tests focused** - one concept per test
7. **Use appropriate fixtures** - leverage shared setup in conftest.py

This organized structure ensures our test suite remains maintainable, efficient, and reliable as the project grows.