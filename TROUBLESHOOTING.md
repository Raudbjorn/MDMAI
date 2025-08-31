# MDMAI Development Troubleshooting Guide

This document addresses common issues encountered during development and testing of the MDMAI project.

## Quick Fix Script

For most setup issues, run the automated setup script:

```bash
chmod +x dev_setup.sh
./dev_setup.sh
```

## Common Issues and Solutions

### Issue 1: Git Path Problems with Frontend Files

**Problem**: Getting errors like `could not open directory 'frontend/frontend/src/lib'` or `pathspec did not match any files`

**Solution**:
- Ensure you're in the project root directory (`/path/to/MDMAI`)
- Frontend files are located at `frontend/src/lib/...` (not `frontend/frontend/...`)
- Use `git status` to see the correct paths for staging files

```bash
# Correct way to add frontend files
git add frontend/src/lib/components/collaboration/ParticipantList.svelte
# Or from frontend directory:
cd frontend && git add src/lib/components/collaboration/ParticipantList.svelte
```

### Issue 2: Virtual Environment Issues

**Problem**: Virtual environments not found or import errors

**Solutions**:
1. **Clean Setup**: Remove old environments and create fresh ones
   ```bash
   rm -rf venv test_venv .venv
   # Try python3 first, fallback to python if python3 is not available
   python3 -m venv venv || python -m venv venv
   source venv/bin/activate
   pip install --upgrade pip setuptools wheel
   ```

2. **Core Dependencies First**: Install essential packages before full requirements
   ```bash
   source venv/bin/activate
   pip install fastmcp==2.11.3 structlog==24.1.0 fastapi==0.109.0
   # Then install full requirements
   pip install -r requirements.txt
   ```

3. **Persistent Activation**: Always activate before running commands
   ```bash
   source venv/bin/activate
   python -c "import fastmcp; print('FastMCP works')"
   ```

### Issue 3: Import Errors (structlog, fastmcp, etc.)

**Problem**: `ModuleNotFoundError: No module named 'structlog'` or similar

**Solutions**:
1. **Check Requirements**: Ensure all dependencies are in requirements.txt
   ```bash
   grep structlog requirements.txt  # Should show: structlog==24.1.0
   ```

2. **Install Missing Packages**:
   ```bash
   source venv/bin/activate
   pip install structlog==24.1.0
   # Or reinstall all requirements
   pip install -r requirements.txt
   ```

3. **Verify Installation**:
   ```bash
   source venv/bin/activate
   python -c "import structlog; print('structlog works')"
   ```

### Issue 4: FastMCP Import/Version Issues

**Problem**: FastMCP import errors or version conflicts

**Solutions**:
1. **Use Correct Version**: Ensure FastMCP 2.11.3 is installed
   ```bash
   pip install fastmcp==2.11.3
   ```

2. **Test Both Import Paths**:
   ```bash
   # New path (preferred)
   python -c "from fastmcp import FastMCP; print('New import works')"
   # Legacy path (used in project)
   python -c "from mcp.server.fastmcp import FastMCP; print('Legacy import works')"
   ```

3. **Check Compatibility**:
   ```bash
   python -c "
   from fastmcp import FastMCP
   mcp = FastMCP('test')
   print(f'FastMCP initialized successfully')
   "
   ```

### Issue 5: Python Linting Errors

**Problem**: Ruff reports hundreds of linting errors

**Solutions**:
1. **Focus on Critical Errors First**: Fix undefined names (F821) as they cause runtime errors
   ```bash
   ruff check src/ --select=F821 --show-source
   ```

2. **Auto-fix What's Possible**:
   ```bash
   ruff check src/ --fix --select=F401,F541,F841
   ```

3. **Manual Fixes for Critical Issues**:
   - F821 (undefined-name): Add missing imports
   - E722 (bare-except): Add specific exception types
   - F401 (unused-import): Remove unused imports

### Issue 6: Frontend TypeScript Errors

**Problem**: Svelte check reports TypeScript errors and accessibility warnings

**Solutions**:
1. **Run Frontend Checks**:
   ```bash
   cd frontend
   npm run check
   ```

2. **Common Fixes**:
   - Add `aria-label` to buttons without text
   - Use `for` attributes to associate labels with form controls
   - Fix TypeScript type mismatches
   - Replace `@apply` directives with inline Tailwind classes

3. **Priority Order**:
   - Fix TypeScript errors first (prevent compilation failures)
   - Fix accessibility warnings second (improve UX)
   - Address style warnings last

### Issue 7: Testing Setup Problems

**Problem**: Tests fail to run due to missing dependencies or configuration

**Solutions**:
1. **Use Convenience Scripts**:
   ```bash
   ./run_tests.sh      # Run Python tests
   ./run_linting.sh    # Check code quality  
   ./run_typecheck.sh  # Run type checking
   ```

2. **Manual Test Setup**:
   ```bash
   source venv/bin/activate
   pip install pytest pytest-asyncio
   python -m pytest tests/ -v
   ```

3. **Test Individual Components**:
   ```bash
   # Test FastMCP import
   python -c "from fastmcp import FastMCP; print('OK')"
   
   # Test main module
   python -c "from src.main import mcp; print('Main module OK')"
   ```

## Development Best Practices

### Environment Management
- Always use virtual environments
- Keep requirements.txt up to date
- Test imports after adding new dependencies

### Code Quality
- Run linting before committing: `ruff check src/`
- Fix critical errors (F821, E722) immediately
- Use auto-fix for style issues: `ruff check --fix`

### Git Workflow
- Check paths with `git status` before staging
- Work from project root for consistency
- Use relative paths in scripts and documentation

### Frontend Development
- Run `npm run check` before committing
- Fix TypeScript errors before warnings
- Test accessibility with screen readers when possible

## Emergency Reset

If everything breaks, use this nuclear option:

```bash
# Clean everything
rm -rf venv node_modules frontend/node_modules
git clean -fd
git checkout -- .

# Fresh setup
./dev_setup.sh

# Verify
source venv/bin/activate
python -c "import fastmcp, structlog; print('All imports work')"
cd frontend && npm run check
```

## Getting Help

1. Check this troubleshooting guide first
2. Run the `dev_setup.sh` script
3. Check the GitHub issues for similar problems
4. Create a new issue with:
   - Your operating system
   - Python version (`python --version`)
   - Node version (`node --version`)
   - Full error message
   - Steps to reproduce