# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Setup
- Always source the virtual environment before running Python commands: `source .venv/bin/activate`
- This project uses UV as the package manager, not pip
- **Dependency management**: When adding dependencies with UV, use >= constraints (e.g., `uv add "package>=1.2.3"`)
  - Exception: `inspect-ai` must remain pinned to a specific version (0.3.125) for stability
  - Use the latest stable version as the minimum to keep dependencies healthy and secure
  - Check latest versions with `uv pip list --outdated` or on PyPI

## Key Commands

### Development Setup
```bash
# Initial setup
uv venv && uv sync --dev
source .venv/bin/activate

# CRITICAL: Install pre-commit hooks (CI will fail without this!)
pre-commit install
```

### Running Tests
```bash
# Run all unit tests
pytest

# Run integration tests (requires GROQ_API_KEY)
pytest -m integration

# Run specific test file
pytest tests/test_registry.py

# Run with coverage
pytest --cov=openbench --cov-report=term-missing

# Run tests for a specific Python version (CI tests 3.10-3.13)
uv python install 3.12 && uv sync --dev && uv run pytest
```

### Code Quality (MUST pass before committing)
```bash
# Format code
uv run ruff format .

# Lint code with auto-fix
uv run ruff check . --fix

# Type checking
uv run mypy .

# Run all pre-commit hooks (REQUIRED before commit)
pre-commit run --all-files
```

### Using the CLI
```bash
# List available benchmarks (alpha benchmarks shown with --alpha flag)
bench list
bench list --alpha  # Include experimental benchmarks

# Describe a specific benchmark
bench describe mmlu

# Run evaluation (uses BENCH_MODEL env var or --model flag)
bench eval mmlu --model groq/llama-3.3-70b-versatile --limit 10

# Run with model-specific arguments
bench eval mmlu -M reasoning_effort=high -M temperature=0.3

# Run with task-specific arguments
bench eval math -T subject=algebra --model openai/o3-mini

# Retry failed evaluation
bench eval-retry

# View previous results
bench view

# Run local/private evaluation
bench eval /path/to/your/eval.py --model groq/llama-3.3-70b-versatile
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

### Core Architecture Patterns

OpenBench is built on the Inspect AI evaluation framework with a clear separation of concerns:

1. **Task Definition Pattern**: Each benchmark is a decorated function returning an Inspect AI `Task`
   ```python
   @task
   def benchmark_name() -> Task:
       return Task(
           dataset=get_dataset(),
           solver=[system_message(SYSTEM), generate()],
           scorer=robust_mcq_scorer(),
           config=GenerateConfig(temperature=0.5)
       )
   ```

2. **Registry System**: Dynamic loading via `_registry.py`
   - Model providers registered with `@modelapi` decorator
   - Tasks discovered automatically from `evals/` directory
   - No manual registration needed for new benchmarks

3. **Dataset Loading Pattern**: Consistent interface across data sources
   - CSV: `csv_dataset(url, sample_fields=record_to_sample)`
   - JSON: `json_dataset(path, sample_fields=record_to_sample)`
   - HuggingFace: `hf_dataset(repo, split="test")`
   - Memory: `MemoryDataset(samples)` for programmatic datasets

4. **Scorer Architecture**: Reusable scoring components
   - MCQ tasks: `robust_mcq_scorer()` with answer extraction
   - Exact match: `exact()` or `match()` scorers
   - Custom: Create task-specific scorers extending base `Scorer`
   - Metrics: `accuracy()`, `std()`, `stderr()` computed automatically

### Project Structure
```
src/openbench/
├── _cli/              # CLI implementation (typer-based)
│   ├── eval_command.py    # Main eval logic
│   ├── list_command.py    # List benchmarks/models
│   └── view_command.py    # Result viewer
├── datasets/          # Dataset loaders
│   ├── mmlu.py           # Subject-specific MMLU loading
│   ├── humaneval.py      # Code evaluation datasets
│   └── [benchmark].py    # One file per benchmark family
├── evals/            # Benchmark implementations
│   ├── matharena/       # Competition math problems
│   │   ├── aime_*/      # AIME years
│   │   └── hmmt_*/      # HMMT years
│   ├── mmlu.py          # Core knowledge eval
│   └── [benchmark].py   # Task definitions
├── scorers/          # Scoring functions
│   ├── mcq.py           # Multiple choice scoring
│   ├── math.py          # Math answer normalization
│   └── code.py          # Code execution scoring
├── utils/            # Shared utilities
│   ├── text.py          # Text processing, prompts
│   ├── math.py          # Math canonicalization
│   └── regex.py         # Answer extraction patterns
├── model/            # Model provider implementations
│   └── _providers/      # 30+ provider integrations
├── _registry.py      # Dynamic task/model loading
└── config.py        # Benchmark metadata only
```

### Key Implementation Patterns

1. **Multiple Choice Questions (MCQ)**:
   - Use `MULTIPLE_CHOICE_PROMPT_TEMPLATE` for consistent formatting
   - Extract answers with `robust_mcq_scorer()` and `MCQ_PATTERNS`
   - Support multilingual with `MULTILINGUAL_ANSWER_REGEXES`

2. **Math Problems**:
   - Normalize answers with `normalize_number()`, `strip_md_latex()`
   - Use competition-specific scorers for AIME/HMMT
   - Handle LaTeX, fractions, scientific notation

3. **Code Evaluation**:
   - Sandbox execution with Docker containers
   - Test case validation with HumanEval patterns
   - Timeout and resource limits configured

4. **Long Context**:
   - Needle retrieval tasks (MRCR variants)
   - Graph traversal (GraphWalks - alpha)
   - Chunked processing for efficiency

### Adding New Components

#### New Benchmark
1. Create `src/openbench/evals/benchmark_name.py`:
   ```python
   from inspect_ai import Task, task
   from inspect_ai.solver import generate
   from openbench.datasets.benchmark_name import get_dataset
   from openbench.scorers import robust_mcq_scorer

   @task
   def benchmark_name() -> Task:
       """Brief description."""
       return Task(
           dataset=get_dataset(),
           solver=[generate()],
           scorer=robust_mcq_scorer()
       )
   ```

2. Create dataset loader in `src/openbench/datasets/benchmark_name.py`

3. Add metadata to `src/openbench/config.py`:
   ```python
   "benchmark_name": BenchmarkMetadata(
       name="Human Readable Name",
       description="What this benchmark evaluates",
       category="core",  # or "math", "reasoning", etc.
       tags=["tag1", "tag2"],
       module_path="openbench.evals.benchmark_name",
       function_name="benchmark_name"
   )
   ```

#### New Model Provider
1. Create `src/openbench/model/_providers/provider_name.py`
2. Register in `src/openbench/_registry.py`:
   ```python
   @modelapi(name="provider")
   def provider() -> Type[ModelAPI]:
       from .model._providers.provider_name import ProviderAPI
       return ProviderAPI
   ```

### Testing Strategy
- **Unit tests**: Core functionality, scorers, utilities
- **Integration tests**: End-to-end with real API calls (marked with `@pytest.mark.integration`)
- **CI Matrix**: Python 3.10-3.13 tested in parallel
- **Pre-commit**: Enforces code quality before commit

### Important Notes
- Benchmarks requiring graders (SimpleQA, HLE, etc.) need `OPENAI_API_KEY`
- Alpha/experimental benchmarks hidden by default (use `--alpha` flag)
- Results stored in `./logs/` directory as JSON
- Use `bench view` for interactive result exploration
- Model providers need appropriate API keys (GROQ_API_KEY, OPENAI_API_KEY, etc.)