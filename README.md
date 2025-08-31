# OpenBench

**Provider-agnostic, open-source evaluation infrastructure for language models** 🚀

[![PyPI version](https://badge.fury.io/py/openbench.svg)](https://badge.fury.io/py/openbench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

OpenBench provides standardized, reproducible benchmarking for LLMs across 30+ evaluation suites (and growing) spanning knowledge, math, reasoning, coding, science, reading comprehension, health, long-context recall, graph reasoning, and first-class support for your own local evals to preserve privacy. **Works with any model provider** - Groq, OpenAI, Anthropic, Cohere, Google, AWS Bedrock, Azure, local models via Ollama, Hugging Face, and 30+ other providers.

## 🚧 Alpha Release

We're building in public! This is an alpha release - expect rapid iteration. The first stable release is coming soon.

## 🎉 What's New in v0.3.0

- **📡 18 More Model Providers**: Added support for AI21, Baseten, Cerebras, Cohere, Crusoe, DeepInfra, Friendli, Hugging Face, Hyperbolic, Lambda, MiniMax, Moonshot, Nebius, Nous, Novita, Parasail, Reka, SambaNova and more
- **🧪 New Benchmarks**: DROP (reading comprehension), experimental benchmarks available with `--alpha` flag
- **⚡ CLI Enhancements**: `openbench` alias, `-M`/`-T` flags for model/task args, `--debug` mode for eval-retry
- **🔧 Developer Tools**: GitHub Actions integration, Inspect AI extension support

## Features

- **🎯 35+ Benchmarks**: MMLU, GPQA, HumanEval, SimpleQA, competition math (AIME, HMMT), SciCode, GraphWalks, and more
- **🔧 Simple CLI**: `bench list`, `bench describe`, `bench eval` (also available as `openbench`)
- **🏗️ Built on inspect-ai**: Industry-standard evaluation framework
- **📊 Extensible**: Easy to add new benchmarks and metrics
- **🤖 Provider-agnostic**: Works with 30+ model providers out of the box
- **🛠️ Local Eval Support**: Privatized benchmarks can now be run with `bench eval <path>`
- **📤 Hugging Face Integration**: Push evaluation results directly to Hugging Face datasets

## 🏃 Speedrun: Evaluate a Model in 60 Seconds

**Prerequisite**: [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
# Create a virtual environment and install OpenBench (30 seconds)
uv venv
source .venv/bin/activate
uv pip install openbench

# Set your API key (any provider!)
export GROQ_API_KEY=your_key  # or OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.

# Run your first eval (30 seconds)
bench eval mmlu --model groq/llama-3.3-70b-versatile --limit 10

# That's it! 🎉 Check results in ./logs/ or view them in an interactive UI:
bench view
```

https://github.com/user-attachments/assets/e99e4628-f1f5-48e4-9df2-ae28b86168c2

## Using Different Providers

```bash
# Groq (blazing fast!)
bench eval gpqa_diamond --model groq/meta-llama/llama-4-maverick-17b-128e-instruct

# OpenAI
bench eval humaneval --model openai/o3-2025-04-16

# Anthropic
bench eval simpleqa --model anthropic/claude-sonnet-4-20250514

# Google
bench eval mmlu --model google/gemini-2.5-pro

# Local models with Ollama
bench eval musr --model ollama/llama3.1:70b

# Hugging Face Inference Providers
bench eval mmlu --model huggingface/gpt-oss-120b:groq

# 30+ providers supported - see full list below
```

## Supported Providers

OpenBench supports 30+ model providers through Inspect AI. Set the appropriate API key environment variable and you're ready to go:

| Provider              | Environment Variable   | Example Model String             |
| --------------------- | ---------------------- | -------------------------------- |
| **AI21 Labs**         | `AI21_API_KEY`         | `ai21/model-name`                |
| **Anthropic**         | `ANTHROPIC_API_KEY`    | `anthropic/model-name`           |
| **AWS Bedrock**       | AWS credentials        | `bedrock/model-name`             |
| **Azure**             | `AZURE_OPENAI_API_KEY` | `azure/<deployment-name>`        |
| **Baseten**           | `BASETEN_API_KEY`      | `baseten/model-name`             |
| **Cerebras**          | `CEREBRAS_API_KEY`     | `cerebras/model-name`            |
| **Cohere**            | `COHERE_API_KEY`       | `cohere/model-name`              |
| **Crusoe**            | `CRUSOE_API_KEY`       | `crusoe/model-name`              |
| **DeepInfra**         | `DEEPINFRA_API_KEY`    | `deepinfra/model-name`           |
| **Friendli**          | `FRIENDLI_TOKEN`       | `friendli/model-name`            |
| **Google**            | `GOOGLE_API_KEY`       | `google/model-name`              |
| **Groq**              | `GROQ_API_KEY`         | `groq/model-name`                |
| **Hugging Face**      | `HF_TOKEN`             | `huggingface/model-name`         |
| **Hyperbolic**        | `HYPERBOLIC_API_KEY`   | `hyperbolic/model-name`          |
| **Lambda**            | `LAMBDA_API_KEY`       | `lambda/model-name`              |
| **MiniMax**           | `MINIMAX_API_KEY`      | `minimax/model-name`             |
| **Mistral**           | `MISTRAL_API_KEY`      | `mistral/model-name`             |
| **Moonshot**          | `MOONSHOT_API_KEY`     | `moonshot/model-name`            |
| **Nebius**            | `NEBIUS_API_KEY`       | `nebius/model-name`              |
| **Nous Research**     | `NOUS_API_KEY`         | `nous/model-name`                |
| **Novita AI**         | `NOVITA_API_KEY`       | `novita/model-name`              |
| **Ollama**            | None (local)           | `ollama/model-name`              |
| **OpenAI**            | `OPENAI_API_KEY`       | `openai/model-name`              |
| **OpenRouter**        | `OPENROUTER_API_KEY`   | `openrouter/model-name`          |
| **Parasail**          | `PARASAIL_API_KEY`     | `parasail/model-name`            |
| **Perplexity**        | `PERPLEXITY_API_KEY`   | `perplexity/model-name`          |
| **Reka**              | `REKA_API_KEY`         | `reka/model-name`                |
| **SambaNova**         | `SAMBANOVA_API_KEY`    | `sambanova/model-name`           |
| **Together AI**       | `TOGETHER_API_KEY`     | `together/model-name`            |
| **Vercel AI Gateway** | `AI_GATEWAY_API_KEY`   | `vercel/creator-name/model-name` |
| **vLLM**              | None (local)           | `vllm/model-name`                |

## Available Benchmarks

Here are the currently available benchmarks. For an up-to-date list use `bench list`.

> [!NOTE]
> Benchmark names are case-sensitive in the CLI.

| Category          | Benchmarks                                                                                                                                                                                                                                                                 |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Knowledge**     | MMLU (57 subjects), GPQA (graduate-level), SuperGPQA (285 disciplines), OpenBookQA, HLE (Humanity's Last Exam - 2,500 questions from 1,000+ experts), HLE_text (text-only version)                                                                                         |
| **Coding**        | HumanEval (164 problems)                                                                                                                                                                                                                                                   |
| **Math**          | AIME 2023-2025, HMMT Feb 2023-2025, BRUMO 2025, MATH (competition-level problems), MATH-500 (challenging subset), MGSM (multilingual grade school math), MGSM_en (English), MGSM_latin (5 languages), MGSM_non_latin (6 languages)                                         |
| **Reasoning**     | SimpleQA (factuality), MuSR (multi-step reasoning), DROP (discrete reasoning over paragraphs), MMMU (multi-modal reasoning with 30+ subjects)                                                                                                                              |
| **Long Context**  | OpenAI MRCR (multiple needle retrieval), OpenAI MRCR_2n (2 needle), OpenAI MRCR_4 (4 needle), OpenAI MRCR_8n (8 needle)                                                                                                                                                    |
| **Healthcare**    | HealthBench (open-ended healthcare eval), HealthBench_hard (challenging variant), HealthBench_consensus (consensus variant)                                                                                                                                                |
| **Cybersecurity** | CTI-Bench (complete cyber threat intelligence suite), CTI-Bench ATE (MITRE ATT&CK technique extraction), CTI-Bench MCQ (knowledge questions on CTI standards and best practices), CTI-Bench RCM (CVE to CWE vulnerability mapping), CTI-Bench VSP (CVSS score calculation) |

## Configuration

```bash
# Set your API keys
export GROQ_API_KEY=your_key
export HF_TOKEN=your_key
export OPENAI_API_KEY=your_key  # Optional

# Set default model
export BENCH_MODEL=groq/llama-3.1-70b
```

## Commands and Options

For a complete list of all commands and options, run: `bench --help`

| Command                  | Description                                        |
| ------------------------ | -------------------------------------------------- |
| `bench` or `openbench`   | Show main menu with available commands             |
| `bench list`             | List available evaluations, models, and flags      |
| `bench eval <benchmark>` | Run benchmark evaluation on a model                |
| `bench eval-retry`       | Retry a failed evaluation                          |
| `bench view`             | View logs from previous benchmark runs             |
| `bench eval <path>`      | Run your local/private evals built with Inspect AI |

### Key `eval` Command Options

| Option               | Environment Variable     | Default         | Description                                                      |
| -------------------- | ------------------------ | --------------- | ---------------------------------------------------------------- |
| `-M <args>`          | None                     | None            | Pass model-specific arguments (e.g., `-M reasoning_effort=high`) |
| `-T <args>`          | None                     | None            | Pass task-specific arguments to the benchmark                    |
| `--model`            | `BENCH_MODEL`            | None (required) | Model(s) to evaluate                                             |
| `--epochs`           | `BENCH_EPOCHS`           | `1`             | Number of epochs to run each evaluation                          |
| `--max-connections`  | `BENCH_MAX_CONNECTIONS`  | `10`            | Maximum parallel requests to model                               |
| `--temperature`      | `BENCH_TEMPERATURE`      | `0.6`           | Model temperature                                                |
| `--top-p`            | `BENCH_TOP_P`            | `1.0`           | Model top-p                                                      |
| `--max-tokens`       | `BENCH_MAX_TOKENS`       | `None`          | Maximum tokens for model response                                |
| `--seed`             | `BENCH_SEED`             | `None`          | Seed for deterministic generation                                |
| `--limit`            | `BENCH_LIMIT`            | `None`          | Limit evaluated samples (number or start,end)                    |
| `--logfile`          | `BENCH_OUTPUT`           | `None`          | Output file for results                                          |
| `--sandbox`          | `BENCH_SANDBOX`          | `None`          | Environment to run evaluation (local/docker)                     |
| `--timeout`          | `BENCH_TIMEOUT`          | `10000`         | Timeout for each API request (seconds)                           |
| `--display`          | `BENCH_DISPLAY`          | `None`          | Display type (full/conversation/rich/plain/none)                 |
| `--reasoning-effort` | `BENCH_REASONING_EFFORT` | `None`          | Reasoning effort level (low/medium/high)                         |
| `--json`             | None                     | `False`         | Output results in JSON format                                    |
| `--log-format`       | `BENCH_LOG_FORMAT`       | `eval`          | Output logging format (eval/json)                                |
| `--hub-repo`         | `BENCH_HUB_REPO`         | `None`          | Push results to a Hugging Face Hub dataset                       |

## Grader Information

Some benchmarks use a grader model to score the model's performance. This requires an additional API key for the grader model.

To run these benchmarks, you'll need to export your `OPENAI_API_key`:

```bash
export OPENAI_API_KEY=your_openai_key
```

The following benchmarks use a grader model:

| Benchmark | Default Grader Model |
| :--- | :--- |
| `simpleqa` | `openai/gpt-4.1-2025-04-14` |
| `hle` | `openai/o3-mini-2025-01-31` |
| `hle_text` | `openai/o3-mini-2025-01-31` |
| `browsecomp` | `openai/gpt-4.1-2025-04-14` |
| `healthbench` | `openai/gpt-4.1-2025-04-14` |
| `math` | `openai/gpt-4-turbo-preview` |
| `math_500` | `openai/gpt-4-turbo-preview` |

## Building Your Own Evals

OpenBench is built on [Inspect AI](https://inspect.aisi.org.uk/). To create custom evaluations, check out their excellent [documentation](https://inspect.aisi.org.uk/). Once you do build your own private evaluations with Inspect AI that you don't want to open-source, you can point OpenBench at them with `bench eval <path>` to run!

## Exporting Logs to Hugging Face

OpenBench can export logs to a Hugging Face Hub dataset. This is useful if you want to share your results with the community or use them for further analysis.

```bash
export HF_TOKEN=<your-huggingface-token>

bench eval mmlu --model groq/llama-3.3-70b-versatile --limit 10 --hub-repo <your-username>/openbench-logs
```

This will export the logs to a Hugging Face Hub dataset with the name `openbench-logs`.

## FAQ

### How does OpenBench differ from Inspect AI?

OpenBench provides:

- **Reference implementations** of 20+ major benchmarks with consistent interfaces
- **Shared utilities** for common patterns (math scoring, multi-language support, etc.)
- **Curated scorers** that work across different eval types
- **CLI tooling** optimized for running standardized benchmarks

Think of it as a benchmark library built on Inspect's excellent foundation.

### Why not just use Inspect AI, lm-evaluation-harness, or lighteval?

Different tools for different needs! OpenBench focuses on:

- **Shared components**: Common scorers, solvers, and datasets across benchmarks reduce code duplication
- **Clean implementations**: Each eval is written for readability and reliability
- **Developer experience**: Simple CLI, consistent patterns, easy to extend

We built OpenBench because we needed evaluation code that was easy to understand, modify, and trust. It's a curated set of benchmarks built on Inspect AI's excellent foundation.

### How can I run `bench` outside of the `uv` environment?

If you want `bench` to be available outside of `uv`, you can run the following command:

```bash
uv run pip install -e .
```

### I'm running into an issue when downloading a dataset from HuggingFace - how do I fix it?

Some evaluations may require logging into HuggingFace to download the dataset. If `bench` prompts you to do so, or throws "gated" errors,
defining the environment variable

```bash
HF_TOKEN="<HUGGINGFACE_TOKEN>"
```

should fix the issue. The full HuggingFace documentation can be found [on the HuggingFace docs on Authentication](https://huggingface.co/docs/hub/en/datasets-polars-auth).

## Development

For development work, you'll need to clone the repository:

```bash
# Clone the repo
git clone https://github.com/groq/openbench.git
cd openbench

# Setup with UV
uv venv && uv sync --dev
source .venv/bin/activate

# CRITICAL: Install pre-commit hooks (CI will fail without this!)
pre-commit install

# Run tests
pytest
```

⚠️ **IMPORTANT**: You MUST run `pre-commit install` after setup or CI will fail on your PRs!

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for detailed instructions on:

- Setting up the development environment
- Adding new benchmarks and model providers
- Code style and testing requirements
- Submitting issues and pull requests

Quick links:

- [Report a bug](https://github.com/groq/openbench/issues/new?assignees=&labels=bug&projects=&template=bug_report.yml)
- [Request a feature](https://github.com/groq/openbench/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.yml)

## Reproducibility Statement

As the authors of OpenBench, we strive to implement this tool's evaluations as faithfully as possible with respect to the original benchmarks themselves.

However, it is expected that developers may observe numerical discrepancies between OpenBench's scores and the reported scores from other sources.

These numerical differences can be attributed to many reasons, including (but not limited to) minor variations in the model prompts, different model quantization or inference approaches, and repurposing benchmarks to be compatible with the packages used to develop OpenBench.

As a result, OpenBench results are meant to be compared with OpenBench results, not as a universal one-to-one comparison with every external result. For meaningful comparisons, ensure you are using the same version of OpenBench.

We encourage developers to identify areas of improvement and we welcome open source contributions to OpenBench.

## Acknowledgments

This project would not be possible without:

- **[Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai)** - The incredible evaluation framework that powers OpenBench
- **[EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)** - Pioneering work in standardized LLM evaluation
- **[Hugging Face's lighteval](https://github.com/huggingface/lighteval)** - Excellent evaluation infrastructure

## Citation

```bibtex
@software{openbench,
  title = {OpenBench: Provider-agnostic, open-source evaluation infrastructure for language models},
  author = {Sah, Aarush},
  year = {2025},
  url = {https://openbench.dev}
}
```

## License

MIT

---

Built with ❤️ by [Aarush Sah](https://github.com/AarushSah) and the [Groq](https://groq.com) team
