# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Setup
- Always source the virtual environment before running Python commands: `source .venv/bin/activate`
- This project uses UV as the package manager, not pip
- **Dependency management**: When adding dependencies with UV, use >= constraints (e.g., `uv add "package>=1.2.3"`)
  - Exception: `inspect-ai` must remain pinned to a specific version for stability
  - Use the latest stable version as the minimum to keep dependencies healthy and secure
  - Check latest versions with `uv pip list --outdated` or on PyPI

## Key Commands

### Development Setup
```bash
# Initial setup
uv venv && uv sync --dev
source .venv/bin/activate
```

### Running Tests
```bash
# Run all unit tests
pytest

# Run integration tests (requires API keys)
pytest -m integration

# Run specific test file
pytest tests/test_registry.py

# Run with coverage
pytest --cov=openbench
```

### Code Quality
```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy .

# Run all pre-commit hooks
pre-commit run --all-files
```

### Using the CLI
```bash
# List available benchmarks
bench list

# Describe a specific benchmark
bench describe mmlu

# Run evaluation
bench eval mmlu --model groq/llama-3.1-70b --limit 10

# View previous results
bench view
```

### Installing Dependencies
```bash
# Install core dependencies only (runs most benchmarks)
uv sync

# Install with specific benchmark dependencies
uv sync --group scicode       # For SciCode benchmark
uv sync --group jsonschemabench # For JSONSchemaBench

# Install with development tools
uv sync --dev

# Install everything
uv sync --all-groups
```

### Publishing to PyPI
```bash
# Build the package
uv build

# Publish to PyPI (requires PyPI API token)
uv publish
```

## Architecture Overview

### Project Structure
- `src/openbench/` - Main package directory
  - `_cli/` - CLI commands implementation
  - `datasets/` - Dataset loaders (MMLU, GPQA, HumanEval, SimpleQA)
  - `evals/` - Benchmark implementations built on Inspect AI
  - `metrics/` - Custom scoring metrics
  - `scorers/` - Scoring functions used across benchmarks
  - `utils/` - Shared utilities
  - `_registry.py` - Dynamic task loading system
  - `config.py` - Benchmark metadata and configuration

### Key Patterns
1. **Registry-based loading**: Benchmarks are dynamically loaded via `_registry.py`
2. **Inspect AI framework**: All evaluations extend Inspect AI's task/solver pattern
3. **Provider-agnostic**: Uses Inspect AI's model abstraction for 15+ providers
4. **Shared components**: Common scorers and utilities reduce code duplication

### Dependency Architecture
OpenBench uses a tightly coupled architecture where benchmarks share common infrastructure:

```
┌─────────────┐      ┌──────────────┐      ┌──────────────┐
│   EVALS     │ ---> │   DATASETS   │ <--> │   SCORERS    │
│  (19 files) │      │  (shared)    │      │  (shared)    │
└─────────────┘      └──────────────┘      └──────────────┘
```

**Core Dependencies** (in main `dependencies`):
- `inspect-ai`: Required by all benchmarks
- `scipy`, `numpy`: Used by multiple scorers (DROP, MMLU, HealthBench)
- `datasets`, `typer`, `groq`, `openai`: Core infrastructure

**Optional Dependencies** (in `dependency-groups`):
- `scicode`: Only needed for SciCode benchmark
- `jsonschemabench`: Only needed for JSONSchemaBench

Most benchmarks (17/19) can run with just core dependencies. Only specialized benchmarks require their specific dependency group.

### Adding New Benchmarks
1. Create evaluation file in `src/openbench/evals/`
2. Add metadata to `config.py`
3. Follow existing patterns for dataset loading and scoring

## Development Standards
- Use type hints for all functions
- Follow conventional commits (feat, fix, docs, etc.)
- Line length: 88 characters (configured in ruff)
- Test new features with both unit and integration tests
- **IMPORTANT: All pre-commit hooks MUST pass before committing**
  - Run `pre-commit run --all-files` to check all hooks
  - The configured hooks are:
    1. `ruff check --fix` - Linting with auto-fix
    2. `ruff format` - Code formatting
    3. `mypy .` - Type checking
  - If any hook fails, fix the issues and re-run until all pass