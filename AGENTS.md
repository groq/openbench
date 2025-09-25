# Repository Guidelines

## Quick Start Setup
1. Clone and enter the repo: `git clone https://github.com/groq/openbench.git && cd openbench`.
2. Create the environment and install tooling: `uv venv && uv sync --dev`.
3. Activate the virtualenv for every session: `source .venv/bin/activate`.
4. Install required git hooks (CI enforces them): `pre-commit install`.
5. Verify the baseline: `pytest` followed by `pre-commit run --all-files`.

## Project Structure & Module Organization
- `src/openbench/` – main package entry point with typed modules:
  - `_cli/` Typer commands (`bench list`, `bench eval`, etc.).
  - `agents/` provider and code-agent adapters.
  - `datasets/`, `evals/`, `metrics/`, `scorers/`, `solvers/`, `utils/` – reusable benchmark building blocks.
  - `_registry.py` dynamic task registration; `config.py` benchmark metadata; `provider_config.py` provider wiring.
- `tests/` mirrors the package layout; networked suites live in `tests/integration/`.
- `docs/` houses Mintlify documentation; `dist/` stores build artifacts; `logs/` captures evaluation runs; `scripts/` holds maintenance helpers.

## Environment & Dependency Management
- UV is the single source of truth; avoid bare `pip install`.
- Add dependencies with `uv add "package>=x.y.z"`; keep `inspect-ai` pinned to the existing exact version.
- Optional dependency groups: `uv sync --group scicode`, `--group jsonschemabench`, or `--all-groups` when you need every benchmark.
- To refresh tooling, run `uv sync --dev` or check for outdated packages with `uv pip list --outdated`.
- Use a local `.env` (untracked) for secrets; never commit credentials.

## Build, Test & Quality Commands
- Unit tests: `pytest` or target a module (`pytest tests/test_registry.py`).
- Integration tests (require provider keys): `pytest -m integration`.
- Coverage guardrail: `pytest --cov=src/openbench` (CI expects coverage parity).
- Quality gates (pre-commit runs these):
  - `ruff check .` and `ruff format .` for linting/formatting.
  - `mypy .` for type safety.
  - `pre-commit run --all-files` to replay the full hook chain.

## Coding Standards & Naming Conventions
- PEP 8, four-space indentation, and 88-character line width (ruff enforces).
- Modules/functions use `snake_case`; classes use `PascalCase`; constants stay `UPPER_SNAKE_CASE`.
- Match CLI file names to command names inside `_cli/`.
- Annotate new public functions with type hints and keep docstrings focused on behavior.
- Prefer `ruff` autofixes over manual formatting; rerun hooks after edits.

## Testing Guidelines
- Place new tests next to the code they exercise (e.g., `tests/metrics/test_new_metric.py`).
- Name files `test_<area>.py` and mark slower or provider-backed cases with `@pytest.mark.integration`.
- Reuse fixtures in `tests/conftest.py`; store golden assets in `tests/fixtures/`.
- For instrumentation or metrics changes, run full coverage and confirm no regression.

## CLI Workflow & Evaluation Practices
- Discover tasks with `bench list` and inspect details via `bench describe <benchmark>`.
- Run evaluations using `bench eval <benchmark> --model <provider/model> --limit 10`; pass task args with `-T` and model flags with `-M`.
- Graded benchmarks (e.g., `simpleqa`, `hle`, `healthbench`) require `OPENAI_API_KEY` for default grader models; export keys before running.
- Use `bench view` to inspect historical logs in `./logs/`, and `--hub-repo <user/repo>` to push results to Hugging Face (needs `HF_TOKEN`).

## Provider Configuration & Secrets
- Core provider tokens: `GROQ_API_KEY`, `HF_TOKEN`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, etc.; check `README.md` for the full matrix.
- Update `provider_config.py` when introducing new providers or auth flows, and document changes under `docs/`.
- Never hardcode secrets; rely on env vars or local `.env` files kept out of git.

## Architecture & Extensibility
- New benchmarks live in `src/openbench/evals/` with any supporting datasets, scorers, or metrics placed in their respective modules.
- Register tasks in `_registry.py` and extend metadata in `config.py`.
- Reuse shared utilities whenever possible; Inspect AI patterns (tasks, solvers, scorers) keep implementations consistent.
- For code agents and Exercism tasks, ensure `agents/` and CLI parameters accept the new option.

## Commit & Pull Request Guidelines
- Use Conventional Commits (`type(scope): summary`) as seen in history (`feat(openrouter): ...`, `chore(docs): ...`).
- Keep each PR single-purpose; discuss complex proposals via issues before implementation.
- Before pushing, run `pytest`, `pytest -m integration` when relevant, and `pre-commit run --all-files`.
- Update docs or changelog entries when behavior changes, attach CLI output or screenshots for UX updates, and link related issues.
- PRs are squash-merged, so set the title to the final Conventional Commit message.

## Release & Publishing Notes
- Build packages with `uv build`; artifacts appear under `dist/`.
- Maintainers publish via `uv publish` using a configured PyPI token.
- Confirm version bumps and changelog updates before release; tag releases to keep evaluations reproducible.

## Further References
- `README.md` – feature overview, CLI options, and provider matrix.
- `CONTRIBUTING.md` – deep dive on workflow expectations and architecture patterns.
- `docs/` – Mintlify site content for end users; update alongside code changes impacting documentation.
