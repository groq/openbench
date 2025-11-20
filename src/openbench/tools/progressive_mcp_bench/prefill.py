import os
import asyncio
import sys
from pathlib import Path
from inspect_ai import Task, eval
from inspect_ai.solver import solver
from inspect_ai.agent import react, AgentPrompt
from inspect_ai.model import GenerateConfig
from inspect_ai.dataset import Sample, MemoryDataset
from openbench.tools.livemcpbench.copilot.toolsource import copilot_tool_source
from openbench.tools.livemcpbench.copilot.arg_generation import McpArgGenerator
from openbench.tools.progressive_mcp_bench.discover_tools import generate_tools_json
from openbench.utils.text import LIVEMCPBENCH_SYSTEM_MESSAGE

# Constants
DATA_DIR = Path(__file__).resolve().parents[2] / "datasets/progressive_mcp_bench/data"
SERVERS_CONFIG = Path(__file__).resolve().parents[2] / "datasets/progressive_mcp_bench/servers.json"
MCP_ROOT = Path(os.path.expanduser("~/.openbench/progressive_mcp_bench"))
DB_PATH = MCP_ROOT / "progressive_mcpbench.sqlite"
MCP_DATA_PATH = MCP_ROOT / "config/mcp_arg.json"
TOOLS_JSON_PATH = MCP_ROOT / "config/tools.json"

def ensure_paths():
    MCP_ROOT.mkdir(parents=True, exist_ok=True)
    (MCP_ROOT / "config").mkdir(parents=True, exist_ok=True)

def prepare_environment():
    ensure_paths()
    
    # 1. Discover tools from servers.json
    if not TOOLS_JSON_PATH.exists():
        print("Discovering tools...")
        asyncio.run(generate_tools_json(SERVERS_CONFIG, TOOLS_JSON_PATH))
    
    # 2. Generate embeddings
    if not MCP_DATA_PATH.exists():
        print("Generating embeddings...")
        # if not os.environ.get("OPENAI_API_KEY"):
        #      raise RuntimeError("OPENAI_API_KEY required for embedding generation")
             
        generator = McpArgGenerator(config=TOOLS_JSON_PATH, output_file=MCP_DATA_PATH)
        asyncio.run(generator.generate())

def get_samples():
    ppt_path = DATA_DIR / "ppt/build_effective_agents.pptx"
    word_path = DATA_DIR / "word/exchange.docx"
    
    return [
        Sample(
            id="ppt_1",
            input=f"What is the title of {ppt_path}?",
            metadata={"server": "ppt"}
        ),
        Sample(
            id="word_1",
            input=f"Summarize the document at {word_path}",
            metadata={"server": "word"}
        ),
        Sample(
            id="playwright_1",
            input="Go to https://example.com and tell me the title of the page.",
            metadata={"server": "playwright"}
        ),
    ]

@solver
def prefill_solver(db_path: Path, model_name: str):
    async def solve(state, generate):
        task_id = state.sample_id
        
        # Env vars for the server process
        extra_env = {
            "PROGRESSIVE_MCP_DB": str(db_path),
            "PROGRESSIVE_MCP_RECORD": "1",
            "PROGRESSIVE_MCP_TASK_ID": str(task_id),
            "PROGRESSIVE_MCP_MODEL": model_name,
            "MCP_SERVERS_CONFIG": str(SERVERS_CONFIG),
            "MCP_DATA_PATH": str(MCP_DATA_PATH),
            "OPENBENCH_COPILOT_AUTOGEN": "1",
            "OPENBENCH_COPILOT_NO_EMBEDDINGS": "1",
        }
        
        tool_source = copilot_tool_source(
            extra_env=extra_env,
            python_executable=sys.executable
        )
        
        agent = react(
            prompt=AgentPrompt(
                instructions=LIVEMCPBENCH_SYSTEM_MESSAGE,
            ),
            tools=[tool_source],
        )
        
        return await agent(state)
        
    return solve

def run_prefill(model="openai/gpt-4o"):
    prepare_environment()
    samples = get_samples()
    
    dataset = MemoryDataset(samples)
    
    task = Task(
        dataset=dataset,
        solver=[prefill_solver(DB_PATH, model)],
        name="progressive_mcp_prefill",
        config=GenerateConfig(
            temperature=0.0, # Deterministic
        )
    )
    
    print(f"Running pre-fill with model {model}...")
    print(f"Database: {DB_PATH}")
    
    results = eval(task, model=model)
    
    # Extract and save expectations
    expectations = {}
    if results and results[0].samples:
        for sample in results[0].samples:
            if sample.output.completion:
                expectations[sample.id] = sample.output.completion
    
    import json
    with open(MCP_ROOT / "expectations.json", "w") as f:
        json.dump(expectations, f, indent=2)
        
    print(f"Saved {len(expectations)} expectations to {MCP_ROOT}/expectations.json")
    
    return results

if __name__ == "__main__":
    # You can run this directly
    run_prefill()
