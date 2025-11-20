# ProgressiveMCPBench

**ProgressiveMCPBench** is a benchmark designed to quantitatively measure an LLM agent's ability to handle "tool clutter." It evaluates how well models perform when presented with varying degrees of irrelevant tools, ranging from a minimal set to a massive library of distracting tools.

## Design Philosophy

Traditionally, agents are given a specific set of tools required for a task. However, in real-world "OS Copilot" scenarios, an agent might have access to hundreds of tools across dozens of applications (Word, PowerPoint, Browser, Email, etc.).

**ProgressiveMCPBench** tests the hypothesis that "more tools = more confusion." It isolates this variable by:
1.  **Freezing the World**: Pre-recording interactions with real tools so the evaluation itself is deterministic and network-independent.
2.  **Progressive Disclosure**: Running the *exact same tasks* against the *exact same frozen data* but with different tool visibility strategies.

## The Pre-fill Step

The core of this benchmark is the **Pre-fill** process. This is a one-time data generation step that "records" the world.

### Purpose
*   To generate a "Golden Path" solution using a high-intelligence model (the "Oracle").
*   To capture the inputs and outputs of real MCP servers (PowerPoint, Word, Playwright) so they can be replayed later without running the actual applications.
*   To discover the full universe of available tools.

### Properties
*   **High Intelligence**: Uses a top-tier model (e.g., GPT-4o, Claude 3.5 Sonnet) to ensure the task is solved correctly and efficient tools are selected.
*   **Real Execution**: The pre-fill agent connects to *actual* running MCP servers and performs real actions (opening files, browsing the web).
*   **VCR Recording**: Every `route` decision and `execute-tool` result is serialized into a SQLite database (`progressive_mcpbench.sqlite`).
*   **Golden Answer**: The final output of the pre-fill run is saved as the expected answer for the benchmark.

### Usage
```bash
# Requires OPENAI_API_KEY for the Oracle model
python -m openbench.tools.progressive_mcp_bench.prefill
```

## Evaluation Strategies

During evaluation, the agent interacts with a **Mock MCP Server** backed by the SQLite database. The server behaves differently depending on the chosen strategy:

### 1. Minimal (`progressive_mcp_bench_minimal`)
*   **Description**: The agent is exposed *only* to the specific tools that were successfully used by the Oracle during the pre-fill step.
*   **Difficulty**: Easy.
*   **Goal**: Tests basic tool usage competency. "If I hand you the exact screwdriver you need, can you turn the screw?"

### 2. All-Relevant (`progressive_mcp_bench_relevant`)
*   **Description**: The agent is exposed to all tools from the *relevant* MCP server for the task. For example, if the task is "summarize this PPT," the agent sees all ~30 PowerPoint tools, but no Word or Playwright tools.
*   **Difficulty**: Medium.
*   **Goal**: Tests domain-specific filtering. "Can you find the right tool within the correct application?"

### 3. All (`progressive_mcp_bench_all`)
*   **Description**: The agent is exposed to the entire universe of tools discovered during pre-fill (potentially hundreds).
*   **Difficulty**: Hard.
*   **Goal**: Tests robustness against noise. "Can you find the right tool in a messy, crowded toolbox without getting distracted?"

## Architecture

*   **`prefill.py`**: Driver script that runs the Oracle agent and populates the DB.
*   **`mock_server.py`**: A FastMCP server that intercepts tool calls. It performs a lookup in the SQLite DB to find a matching recorded response. It implements the logic for filtering tool definitions based on the active strategy.
*   **`mock_toolsource.py`**: An `inspect_ai` adapter that spins up the Mock MCP server as a subprocess for each evaluation task.
*   **`recorder.py`**: (In `livemcpbench/copilot`) Middleware that logs traffic to the SQLite DB.

## Running the Benchmark

```bash
# Run the "All" strategy (hardest)
openbench eval progressive_mcp_bench_all --model groq/llama-3.3-70b-versatile

# Run the "Minimal" strategy (easiest)
openbench eval progressive_mcp_bench_minimal --model groq/llama-3.3-70b-versatile
```
