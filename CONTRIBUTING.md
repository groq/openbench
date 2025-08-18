# Contributing to OpenBench

Thank you for your interest in contributing to OpenBench! We welcome contributions from the community and are grateful for your support in making language model evaluation more accessible and reliable.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [UV package manager](https://docs.astral.sh/uv/getting-started/installation/)
- Git

### Setup
```bash
# Clone and setup
git clone https://github.com/groq/openbench.git
cd openbench
uv venv && uv sync --dev
source .venv/bin/activate

# Run tests to verify setup
pytest
```

## 🎯 Core Principles

### Single Responsibility
**Each PR must address a single concern.** This helps us:
- Review changes more effectively
- Maintain a clean git history
- Easily revert changes if needed
- Understand the purpose of each change

Examples of single-concern PRs:
- ✅ Add support for a new benchmark
- ✅ Fix a specific bug in the MMLU scorer
- ✅ Refactor the math canonicalization utility
- ❌ Add new benchmark AND fix unrelated bug
- ❌ Refactor multiple unrelated components

### Clear Separation of Concerns (SoC)
We value clean, modular code with clear boundaries between components:
- Each module should have a single, well-defined purpose
- Avoid tight coupling between components
- Use dependency injection where appropriate
- Keep business logic separate from infrastructure concerns

## 📝 Commit Guidelines

### Conventional Commits
We use [Conventional Commits](https://www.conventionalcommits.org/) for all commit messages. This provides a clear, structured way to communicate changes.

Format: `<type>(<scope>): <subject>`

#### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, missing semicolons, etc.)
- `refactor`: Code refactoring without changing functionality
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `build`: Build system or dependency changes
- `ci`: CI/CD configuration changes
- `chore`: Other changes that don't modify src or test files

#### Examples
```bash
feat(mmlu): add support for MMLU-Pro benchmark
fix(scorer): handle edge case in math canonicalization
docs(readme): update installation instructions
refactor(humaneval): extract common sandbox logic
test(gpqa): add unit tests for diamond scorer
perf(eval): optimize parallel sample processing
```

#### Scope
The scope should indicate the component or area affected:
- Benchmark names: `mmlu`, `humaneval`, `gpqa`, etc.
- Components: `cli`, `scorer`, `solver`, `common`
- Infrastructure: `docker`, `ci`, `deps`

### Commit Message Body
For complex changes, add a body to explain:
- What changed and why
- Any breaking changes
- Related issues

Example:
```
feat(aime): add support for AIME 2025 problems

- Add dataset loader for AIME 2025
- Update math scorer to handle new problem formats
- Include official solutions for verification

Closes #123
```

## 🔄 Pull Request Process

### Before You Start
1. Check existing issues and PRs to avoid duplicates
2. For significant changes, open an issue first to discuss
3. Fork the repository and create a feature branch

### Development Workflow
1. **Create a feature branch**
   ```bash
   git checkout -b feat/add-new-benchmark
   ```

2. **Make your changes**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   # Run all tests
   pytest
   
   # Run integration tests (requires API keys)
   pytest -m integration
   
   # Run pre-commit hooks (REQUIRED)
   pre-commit run --all-files
   
   # Test your specific changes
   bench eval <your-benchmark> --limit 5
   ```

4. **Commit with conventional commits**
   ```bash
   git commit -m "feat(benchmark): add support for XYZ benchmark"
   ```

### Submitting Your PR

1. **Push to your fork**
   ```bash
   git push origin feat/add-new-benchmark
   ```

2. **Create a Pull Request**
   - Use a clear, descriptive title following conventional commit format
   - Fill out the PR template completely
   - Link any related issues
   - Ensure all CI checks pass

3. **PR Title Format**
   Since we use squash and merge, your PR title becomes the commit message. Use conventional commit format:
   - ✅ `feat(mmlu): add MMLU-Pro support`
   - ✅ `fix(cli): handle missing API key gracefully`
   - ❌ `Updated MMLU benchmark`
   - ❌ `Various fixes`

### What Happens Next
1. A maintainer will review your PR
2. Address any feedback or requested changes
3. Once approved, we'll squash and merge your PR
4. Your contribution will be part of the next release!

## 🏗️ Architecture Guidelines

### Adding a New Benchmark
1. Create a new evaluation file in `src/openbench/evals/`
2. Add dataset loader in `src/openbench/datasets/` if needed
3. Add custom scorer in `src/openbench/scorers/` if needed
4. Register benchmark metadata in `src/openbench/config.py`
5. Use existing utilities from `src/openbench/utils/`
6. Add comprehensive tests

Example structure:
```
src/openbench/
├── evals/
│   └── my_benchmark.py      # Main evaluation logic
├── datasets/
│   └── my_benchmark.py      # Dataset loader
├── scorers/
│   └── my_benchmark.py      # Custom scorer (if needed)
└── config.py                # Add benchmark metadata here
```

### Adding a New Model Provider
1. Create provider file in `src/openbench/model/_providers/`
2. Follow existing provider patterns (see `ai21.py`, `cerebras.py`, etc.)
3. Add environment variable documentation
4. Test with multiple benchmarks
5. Update provider table in README.md

### Key Development Tools
- **UV**: Package manager (not pip) - use `uv add package==version` for dependencies
- **Ruff**: Linting and formatting - replaces Black, isort, flake8
- **MyPy**: Type checking - required for all new code
- **Pre-commit**: Automated code quality checks - must pass before commits
- **Pytest**: Testing framework with integration test markers

### Code Style
- Follow PEP 8 with a line length of 88 characters (Black default)
- Use type hints for all function signatures
- Write docstrings for all public functions and classes
- Prefer composition over inheritance
- Keep functions small and focused

### Testing
- Write unit tests for all new functionality
- Include integration tests for new benchmarks
- Aim for high test coverage
- Test edge cases and error conditions

## 🐛 Reporting Issues

We have structured issue templates to help you report problems effectively:

### Bug Reports
Use our [bug report template](https://github.com/groq/openbench/issues/new?assignees=&labels=bug&projects=&template=bug_report.yml) which includes:
- OpenBench version and environment details
- Exact command that failed
- Expected vs actual behavior
- Error logs and reproduction steps

### Feature Requests
Use our [feature request template](https://github.com/groq/openbench/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.yml) for:
- New benchmarks/evaluations
- New model providers
- CLI enhancements
- Performance improvements
- API/SDK features
- Integration requests

## 📚 Resources

- [Inspect AI Documentation](https://inspect.aisi.org.uk/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [pytest Documentation](https://docs.pytest.org/)

## 🤝 Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to:
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Respect differing viewpoints and experiences

## 📄 License

By contributing to OpenBench, you agree that your contributions will be licensed under the MIT License.