# openbench

<h2 align="center">
 <br>
 <img src="docs/logo/dark.svg" alt="openbench" width="250">
 <br>
 <br>
Provider-agnostic, open-source evaluation infrastructure for language models
 <br>
</h2>

<p align="center">
 <a href="https://badge.fury.io/py/openbench"><img src="https://badge.fury.io/py/openbench.svg">
 <a href="https://github.com/groq/openbench/blob/main/LICENSE.md"><img src="https://img.shields.io/badge/License-MIT-yellow.svg">
 <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg">
 <a href="https://github.com/groq/openbench/stargazers"><img src="https://img.shields.io/github/stars/groq/openbench"></a>
 </a>
</p>


openbench provides standardized, reproducible benchmarking for LLMs across 30+ evaluation suites (and growing) spanning knowledge, math, reasoning, coding, science, reading comprehension, health, long-context recall, graph reasoning, and first-class support for your own local evals to preserve privacy. **Works with any model provider** - Groq, OpenAI, Anthropic, Cohere, Google, AWS Bedrock, Azure, local models via Ollama, Hugging Face, and 30+ other providers.

To get started, see the tutorial below or reference the [docs](https://openbench.dev/).

## Features

- **🎯 95+ Benchmarks**: MMLU, GPQA, HumanEval, SimpleQA, competition math (AIME, HMMT), SciCode, GraphWalks, and more
- **🔧 Simple CLI**: `bench list`, `bench describe`, `bench eval` (also available as `openbench`), `-M`/`-T` flags for model/task args, `--debug` mode for eval-retry, experimental benchmarks with `--alpha` flag
- **🏗️ Built on inspect-ai**: Industry-standard evaluation framework
- **📊 Extensible**: Easy to add new benchmarks and metrics
- **🤖 Provider-agnostic**: Works with 30+ model providers out of the box
- **🛠️ Local Eval Support**: Privatized benchmarks can be run with `bench eval <path>`
- **📤 Hugging Face Integration**: Push evaluation results directly to Hugging Face datasets

## 🏃 Speedrun: Evaluate a Model in 60 Seconds

**Prerequisite**: [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
# Create a virtual environment and install openbench (30 seconds)
uv venv
source .venv/bin/activate
uv pip install openbench

# Set your API key (any provider!)
export GROQ_API_KEY=your_key  # or OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.

# Run your first eval (3 seconds)
bench eval mmlu --model groq/openai/gpt-oss-120b --limit 10

# That's it! 🎉 Check results in ./logs/ or view them in an interactive UI:
bench view
```

https://github.com/user-attachments/assets/e99e4628-f1f5-48e4-9df2-ae28b86168c2

## Supported Providers

openbench supports 30+ model providers through Inspect AI. Set the appropriate API key environment variable and you're ready to go:

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
| **Helicone**          | `HELICONE_API_KEY`     | `helicone/model-name`            |
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
| **SiliconFlow**       | `SILICONFLOW_API_KEY`  | `siliconflow/model-name`         |
| **Together AI**       | `TOGETHER_API_KEY`     | `together/model-name`            |
| **Vercel AI Gateway** | `AI_GATEWAY_API_KEY`   | `vercel/creator-name/model-name` |
| **W&B Inference**     | `WANDB_API_KEY`        | `wandb/model-name`               |
| **vLLM**              | None (local)           | `vllm/model-name`                |

## Available Benchmarks

See the [Benchmarks Catalog](https://openbench.dev/benchmarks/catalog) or use `bench list`.


## Commands and Options

For a complete list of all commands and options, run: `bench --help`
See the [docs](https://openbench.dev/cli/overview) for more details. 

| Command                  | Description                                        |
| ------------------------ | -------------------------------------------------- |
| `bench list`             | List available benchmarks                          |
| `bench eval <benchmark>` | Run benchmark evaluation                           |
| `bench eval-retry <log_files>`  | Retry a failed evaluation                          |
| `bench view`             | Interactive UI to view benchmark logs             |
| `bench cache <info/ls/clear/upload>`            | Manage OpenBench caches            |


### Common `eval` Configuration Options
| Option               | Environment Variable      | Default                   | Description                                                      |
| -------------------- |---------------------------|---------------------------|------------------------------------------------------------------|
| `-M <args>`          | `None`                      | `None`                      | Pass provider/model-specific arguments (e.g., `-M only=groq`)    |
| `-T <args>`          | `None`                      | `None`                      | Pass task-specific arguments to the benchmark                    |
| `--model`            | `BENCH_MODEL`             | `groq/openai/gpt-oss-20b` | Model(s) to evaluate                                             |
| `--epochs`           | `BENCH_EPOCHS`            | `1`                       | Number of epochs to run each evaluation                          |
| `--epochs-reducer`   | `BENCH_EPOCHS_REDUCER`    | `None`                      | Reducer(s) applied when aggregating epoch scores                |
| `--max-connections`  | `BENCH_MAX_CONNECTIONS`   | `10`                      | Maximum parallel requests to model                               |
| `--temperature`      | `BENCH_TEMPERATURE`       | `0.6`                     | Model temperature                                                |
| `--top-p`            | `BENCH_TOP_P`             | `1.0`                     | Model top-p                                                      |
| `--max-tokens`       | `BENCH_MAX_TOKENS`        | `None`                    | Maximum tokens for model response                                |
| `--seed`             | `BENCH_SEED`              | `None`                    | Seed for deterministic generation                                |
| `--limit`            | `BENCH_LIMIT`             | `None`                    | Limit evaluated samples (number or start,end)                    |
| `--logfile`          | `BENCH_OUTPUT`            | `None`                    | Output file for results                                          |
| `--sandbox`          | `BENCH_SANDBOX`           | `None`                    | Environment to run evaluation (local/docker)                     |
| `--timeout`          | `BENCH_TIMEOUT`           | `10000`                   | Timeout for each API request (seconds)                           |
| `--fail-on-error`          | `None`           | `1`                   | Threshold of allowable sample errors (use an integer for count or a float for proportion) |
| `--display`          | `BENCH_DISPLAY`           | `None`                    | Display type (full/conversation/rich/plain/none)                 |
| `--reasoning-effort` | `BENCH_REASONING_EFFORT`  | `None`                    | Reasoning effort level (low/medium/high)                         |
| `--json`             | `None`                      | `False`                   | Output results in JSON format                                    |
| `--log-format`       | `BENCH_LOG_FORMAT`        | `eval`                    | Output logging format (eval/json)                                |
| `--hub-repo`         | `BENCH_HUB_REPO`          | `None`                    | Push results to a Hugging Face Hub dataset                       |
| `--keep-livemcp-root` | `BENCH_KEEP_LIVEMCP_ROOT` | `False`                   | Allow preservation of root data after livemcpbench eval runs     |
| `--code-agent`       | `BENCH_CODE_AGENT`        | `opencode`                | Select code agent for exercism tasks                             |


## Development and Building Your Own Evals

For a full guide, see [Contributing Guidelines](CONTRIBUTING.md) and [Extending openbench](https://openbench.dev/development/extending). Also, check out Inspect AI's excellent [documentation](https://inspect.aisi.org.uk/).

### Quick Eval: Run from Path

For one-off or private evaluations, point openbench directly at your eval:

```bash
bench eval /path/to/my_eval.py --model groq/llama-3.3-70b-versatile
```

### Plugin System: Distribute as Packages

openbench supports a **plugin system via Python entry points**. Package your benchmarks and distribute them independently:

```toml
# pyproject.toml
[project.entry-points."openbench.benchmarks"]
my_benchmark = "my_pkg.metadata:get_benchmark_metadata"
```

After `pip install my-benchmark-package`, your benchmark appears in `bench list` and works with all CLI commands. Perfect for:
- Sharing benchmarks across teams
- Versioning evaluations independently
- Overriding built-in benchmarks with custom implementations

## FAQ

### How does openbench differ from Inspect AI?

openbench provides:

- **Reference implementations** of 20+ major benchmarks with consistent interfaces
- **Shared utilities** for common patterns (math scoring, multi-language support, etc.)
- **Curated scorers** that work across different eval types
- **CLI tooling** optimized for running standardized benchmarks

Think of it as a benchmark library built on Inspect's excellent foundation.

### Why not just use Inspect AI, lm-evaluation-harness, or lighteval?

Different tools for different needs! openbench focuses on:

- **Shared components**: Common scorers, solvers, and datasets across benchmarks reduce code duplication
- **Clean implementations**: Each eval is written for readability and reliability
- **Developer experience**: Simple CLI, consistent patterns, easy to extend

We built openbench because we needed evaluation code that was easy to understand, modify, and trust. It's a curated set of benchmarks built on Inspect AI's excellent foundation.

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

See the docs for further [Tips and Troubleshooting](https://openbench.dev/troubleshooting).

## 🚧 Alpha Release

We're building in public! This is an alpha release - expect rapid iteration. The first stable release is coming soon.

Quick links:

- [Report a bug](https://github.com/groq/openbench/issues/new?assignees=&labels=bug&projects=&template=bug_report.yml)
- [Request a feature](https://github.com/groq/openbench/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.yml)

## Reproducibility Statement

As the authors of openbench, we strive to implement this tool's evaluations as faithfully as possible with respect to the original benchmarks themselves.

However, it is expected that developers may observe numerical discrepancies between openbench's scores and the reported scores from other sources.

These numerical differences can be attributed to many reasons, including (but not limited to) minor variations in the model prompts, different model quantization or inference approaches, and repurposing benchmarks to be compatible with the packages used to develop openbench.

As a result, openbench results are meant to be compared with openbench results, not as a universal one-to-one comparison with every external result. For meaningful comparisons, ensure you are using the same version of openbench.

We encourage developers to identify areas of improvement and we welcome open source contributions to openbench.

## Acknowledgments

This project would not be possible without:

- **[Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai)** - The incredible evaluation framework that powers openbench
- **[EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)** - Pioneering work in standardized LLM evaluation
- **[Hugging Face's lighteval](https://github.com/huggingface/lighteval)** - Excellent evaluation infrastructure

## Citation

```bibtex
@software{openbench,
  title = {openbench: Provider-agnostic, open-source evaluation infrastructure for language models},
  author = {Sah, Aarush},
  year = {2025},
  url = {https://openbench.dev}
}
```

## License

MIT

---

Built with ❤️ by [Aarush Sah](https://github.com/AarushSah) and the [Groq](https://groq.com) team
