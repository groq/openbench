# OpenBench

**Provider-agnostic, open-source evaluation infrastructure for language models** 🚀

[![PyPI version](https://badge.fury.io/py/openbench.svg)](https://badge.fury.io/py/openbench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

OpenBench provides standardized, reproducible benchmarking for LLMs across 20+ evaluation suites (and growing) spanning knowledge, math, reasoning, reading comprehension, health, long-context recall, and first-class support for your own local evals to preserve privacy. **Works with any model provider** - Groq, OpenAI, Anthropic, Cohere, Google, AWS Bedrock, Azure, local models via Ollama, and more.

## 🚧 Alpha Release

We're building in public! This is an alpha release - expect rapid iteration. The first stable release is coming soon.

## Features

- **🎯 30+ Benchmarks**: MMLU, GPQA, HumanEval, SimpleQA, and competition math (AIME, HMMT)
- **🔧 Simple CLI**: `bench list`, `bench describe`, `bench eval`
- **🏗️ Built on inspect-ai**: Industry-standard evaluation framework
- **📊 Extensible**: Easy to add new benchmarks and metrics
- **🤖 Provider-agnostic**: Works with 15+ model providers out of the box
- **🛠️ Local Eval Support**: Privatized benchmarks can now be run with `bench eval <path>`

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

# Any provider supported by Inspect AI!
```
| Category | Benchmarks |
|----------|------------|
| **Knowledge** | MMLU (57 subjects), GPQA (graduate-level), SuperGPQA (285 disciplines), OpenBookQA, HLE (Humanity's Last Exam - 2,500 questions from 1,000+ experts), HLE_text (text-only version) |
| **Coding** | HumanEval (164 problems) |
| **Math** | AIME 2023-2025, HMMT Feb 2023-2025, BRUMO 2025, MATH (competition-level problems), MATH-500 (challenging subset), MGSM (multilingual grade school math), MGSM_en (English), MGSM_latin (5 languages), MGSM_non_latin (6 languages) |
| **Reasoning** | SimpleQA (factuality), MuSR (multi-step reasoning) |
| **Long Context** | OpenAI MRCR (multiple needle retrieval), OpenAI MRCR_2n (2 needle), OpenAI MRCR_4 (4 needle), OpenAI MRCR_8n (8 needle) |
| **Healthcare** | HealthBench (open-ended healthcare eval), HealthBench_hard (challenging variant), HealthBench_consensus (consensus variant) |

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
|--------------------------|----------------------------------------------------|
| `bench`                  | Show main menu with available commands             |
| `bench list`             | List available evaluations, models, and flags      |
| `bench eval <benchmark>` | Run benchmark evaluation on a model                |
| `bench view`             | View logs from previous benchmark runs             |
| `bench eval <path>`      | Run your local/private evals built with Inspect AI |

### Key `eval` Command Options

| Option               | Environment Variable     | Default                                          | Description                                      |
|----------------------|--------------------------|--------------------------------------------------|--------------------------------------------------|
| `--model`            | `BENCH_MODEL`            | `groq/meta-llama/llama-4-scout-17b-16e-instruct` | Model(s) to evaluate                             |
| `--epochs`           | `BENCH_EPOCHS`           | `1`                                              | Number of epochs to run each evaluation          |
| `--max-connections`  | `BENCH_MAX_CONNECTIONS`  | `10`                                             | Maximum parallel requests to model               |
| `--temperature`      | `BENCH_TEMPERATURE`      | `0.6`                                            | Model temperature                                |
| `--top-p`            | `BENCH_TOP_P`            | `1.0`                                            | Model top-p                                      |
| `--max-tokens`       | `BENCH_MAX_TOKENS`       | `None`                                           | Maximum tokens for model response                |
| `--seed`             | `BENCH_SEED`             | `None`                                           | Seed for deterministic generation                |
| `--limit`            | `BENCH_LIMIT`            | `None`                                           | Limit evaluated samples (number or start,end)    |
| `--logfile`          | `BENCH_OUTPUT`           | `None`                                           | Output file for results                          |
| `--sandbox`          | `BENCH_SANDBOX`          | `None`                                           | Environment to run evaluation (local/docker)     |
| `--timeout`          | `BENCH_TIMEOUT`          | `10000`                                          | Timeout for each API request (seconds)           |
| `--display`          | `BENCH_DISPLAY`          | `None`                                           | Display type (full/conversation/rich/plain/none) |
| `--reasoning-effort` | `BENCH_REASONING_EFFORT` | `None`                                           | Reasoning effort level (low/medium/high)         |
| `--json`             | None                     | `False`                                          | Output results in JSON format                    |
| `--hub-repo`         | `BENCH_HUB_REPO`         | `None`                                           | Push results to a Hugging Face Hub dataset                    |

## Building Your Own Evals

OpenBench is built on [Inspect AI](https://inspect.aisi.org.uk/). To create custom evaluations, check out their excellent [documentation](https://inspect.aisi.org.uk/). Once you do build your own private evaluations with Inspect AI that you don't want to open-source, you can point OpenBench  at them with `bench eval <path>` to run!

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

# Run tests
pytest
```

## Contributing

We welcome contributions! Please open issues and PRs at [github.com/groq/openbench](https://github.com/groq/openbench).


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
  title = {OpenBench: Open-source Evaluation Infrastructure for Language Models},
  author = {Sah, Aarush and {Groq Team}},
  year = {2025},
  url = {https://github.com/groq/openbench}
}
```

## License

MIT

---

Built with ❤️ by [Aarush Sah](https://github.com/AarushSah) and the [Groq](https://groq.com) team
