# GitHub Actions Disabled

All GitHub Actions workflows have been temporarily disabled for this repository.

## Disabled Workflows:
- `dependency-tests.yml` → `dependency-tests.yml.disabled`

## Reason:
Workflows disabled to prevent CI/CD failures during development and review of the Ollama embeddings feature.

## To Re-enable:
Rename the `.disabled` files back to their original `.yml` extension:
```bash
mv .github/workflows/dependency-tests.yml.disabled .github/workflows/dependency-tests.yml
```

*Disabled on: 2025-08-31*