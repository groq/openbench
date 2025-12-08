"""
Argument/embedding generation for synthetic MCP servers.

This generates embeddings for the synthetic servers defined in servers.json,
storing them in the same format as the live copilot embeddings.
"""

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm  # type: ignore[import-untyped]

import mcp.types as types
import openai

from openbench.utils.text import LIVEMCPBENCH_TOOL_SUMMARY_PROMPT

load_dotenv()

logger = logging.getLogger(__name__)

embedding_api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
embedding_api_url = os.getenv("EMBEDDING_BASE_URL")

abstract_api_key = os.getenv("ABSTRACT_API_KEY") or os.getenv("OPENAI_API_KEY")
abstract_model = os.getenv("ABSTRACT_MODEL", "gpt-4.1-2025-04-14")
abstract_api_url = os.getenv("ABSTRACT_BASE_URL")


def _synthetic_mcp_dir() -> Path:
    """Get the synthetic_mcp directory path."""
    current = Path(__file__).resolve()
    repo_root = current.parent.parent.parent.parent.parent.parent
    return repo_root / "synthetic_mcp"


def _user_cache_dir() -> Path:
    """Get the user cache directory for copilot embeddings."""
    return Path(
        os.path.expanduser("~/.openbench/progressivemcpbench/copilot")
    ).resolve()


def _per_server_cache_dir() -> Path:
    """Get the directory for per-server embedding cache files."""
    return _user_cache_dir() / "servers" / f"{embedding_model}_{abstract_model}"


def _compute_server_hash(server_name: str, server_config: dict[str, Any]) -> str:
    """Compute a hash for a single server's configuration.

    The hash is based on server name, description, and tool definitions.
    """
    server_data = {
        "name": server_name,
        "description": server_config.get("description", ""),
        "tools": [
            {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
            }
            for tool in server_config.get("tools", [])
        ],
    }
    content = json.dumps(server_data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def _get_server_cache_path(server_name: str, server_hash: str) -> Path:
    """Get the cache file path for a specific server version."""
    return _per_server_cache_dir() / f"{server_name}_{server_hash}.json"


def _combined_embeddings_path() -> Path:
    """Get the path for the combined embeddings file (for consumers)."""
    return _user_cache_dir() / "config" / "synthetic_embeddings_combined.json"


class SyntheticMcpArgGenerator:
    """Generate embeddings for synthetic MCP servers with per-server caching."""

    def __init__(
        self,
        servers_json_path: Path | None = None,
        output_file: Path | None = None,
    ):
        """Initialize the generator.

        Args:
            servers_json_path: Path to synthetic servers.json (default: synthetic_mcp/config/servers.json)
            output_file: Output path for combined embeddings (default: combined cache file)
        """
        self.servers_json_path = servers_json_path or (
            _synthetic_mcp_dir() / "config" / "servers.json"
        )

        if not self.servers_json_path.exists():
            raise FileNotFoundError(
                f"Servers config not found: {self.servers_json_path}"
            )

        with self.servers_json_path.open("r", encoding="utf-8") as f:
            self.servers_config = json.load(f)

        self.output_file = output_file or _combined_embeddings_path()

        self.embedding_client = openai.AsyncOpenAI(
            api_key=embedding_api_key, base_url=embedding_api_url
        )
        self.summary_client = openai.AsyncOpenAI(
            api_key=abstract_api_key, base_url=abstract_api_url
        )

    async def _get_embedding(
        self, text: str, model: str = embedding_model
    ) -> list[float]:
        if not text:
            raise ValueError("Empty text provided for embedding generation")
        response = await self.embedding_client.embeddings.create(
            model=model,
            input=[text],
            encoding_format="float",
        )
        embedding = response.data[0].embedding
        if not embedding or len(embedding) == 0:
            raise ValueError(f"Empty embedding returned for text: {text[:100]}...")
        return embedding

    async def _generate_summary(
        self,
        server_name: str,
        server_desc: str,
        tools: list[types.Tool],
        model: str = abstract_model,
    ) -> str:
        tool_descriptions = "\n".join(
            [f"- {tool.name}: {tool.description}" for tool in tools]
        )

        prompt = LIVEMCPBENCH_TOOL_SUMMARY_PROMPT.format(
            server_name=server_name,
            server_desc=server_desc,
            tool_descriptions=tool_descriptions,
        )
        try:
            response = await self.summary_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert technical writer.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            content = response.choices[0].message.content or ""
            return content.strip()
        except Exception as e:
            logger.error(f"Summary Generation Error for '{server_name}': {e}")
            return f"Error generating summary for {server_name}"

    def _format_tool_parameters(self, tool: types.Tool) -> dict[str, str]:
        formatted_params: dict[str, str] = {}
        schema = tool.inputSchema
        if not schema or "properties" not in schema:
            return formatted_params

        properties = schema.get("properties", {})
        required_params = schema.get("required", [])

        for param_name, param_details in properties.items():
            param_type = param_details.get("type", "any")
            param_desc = param_details.get("description", "")

            if param_name not in required_params:
                formatted_params[param_name] = f"(Optional, {param_type}) {param_desc}"
            else:
                formatted_params[param_name] = f"({param_type}) {param_desc}"
        return formatted_params

    def _get_cached_server_embedding(
        self, server_name: str, server_hash: str
    ) -> dict[str, Any] | None:
        """Load cached embedding for a server if it exists."""
        cache_path = _get_server_cache_path(server_name, server_hash)
        if cache_path.exists():
            try:
                with cache_path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return None

    def _save_server_embedding(
        self, server_name: str, server_hash: str, data: dict[str, Any]
    ) -> None:
        """Save embedding for a server to cache."""
        cache_path = _get_server_cache_path(server_name, server_hash)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    async def _generate_server_embedding(
        self, server_name: str, server_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate embeddings for a single server."""
        server_description = server_config.get("description", "")
        tools_data = server_config.get("tools", [])

        # Convert to types.Tool objects
        tools = []
        for tool_data in tools_data:
            tool = types.Tool(
                name=tool_data.get("name", ""),
                description=tool_data.get("description", ""),
                inputSchema=tool_data.get("inputSchema", {}),
            )
            tools.append(tool)

        # Generate summary
        server_summary = await self._generate_summary(
            server_name, server_description, tools
        )

        # Generate embeddings in parallel
        embedding_tasks = {
            "server_desc": self._get_embedding(server_description),
            "server_summary": self._get_embedding(server_summary),
        }
        for i, tool in enumerate(tools):
            if not tool.description:
                raise ValueError(
                    f"Tool '{tool.name}' in server '{server_name}' has no description"
                )
            embedding_tasks[f"tool_{i}"] = self._get_embedding(tool.description)

        embeddings_results = await asyncio.gather(*embedding_tasks.values())
        embeddings = dict(zip(embedding_tasks.keys(), embeddings_results))

        # Format tools
        formatted_tools = []
        for i, tool in enumerate(tools):
            formatted_tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "description_embedding": embeddings[f"tool_{i}"],
                    "parameter": self._format_tool_parameters(tool),
                }
            )

        return {
            "server_name": server_name,
            "server_summary": server_summary,
            "server_description": server_description,
            "description_embedding": embeddings["server_desc"],
            "summary_embedding": embeddings["server_summary"],
            "tools": formatted_tools,
        }

    async def generate(self, force: bool = False) -> Path:
        """Generate embeddings for all synthetic servers with per-server caching.

        Only servers whose content has changed will be regenerated.

        Args:
            force: If True, regenerate all servers regardless of cache

        Returns:
            Path to the combined embeddings file
        """
        all_servers_info: list[dict[str, Any]] = []
        cached_count = 0
        generated_count = 0

        for server_name, server_config in tqdm(
            self.servers_config.items(), desc="Processing servers"
        ):
            server_hash = _compute_server_hash(server_name, server_config)

            # Check per-server cache
            if not force:
                cached = self._get_cached_server_embedding(server_name, server_hash)
                if cached:
                    all_servers_info.append(cached)
                    cached_count += 1
                    continue

            # Generate new embeddings for this server
            logger.info(f"Generating embeddings for: {server_name}")
            server_output = await self._generate_server_embedding(
                server_name, server_config
            )

            # Cache per-server
            self._save_server_embedding(server_name, server_hash, server_output)
            all_servers_info.append(server_output)
            generated_count += 1

        # Write combined file
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(all_servers_info, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Complete: {cached_count} cached, {generated_count} generated. "
            f"Combined file: {self.output_file}"
        )

        return self.output_file


async def generate_synthetic_embeddings(force: bool = False) -> Path:
    """Generate embeddings for synthetic MCP servers.

    Uses per-server caching - only servers with changed content will be regenerated.

    Args:
        force: If True, regenerate all servers regardless of cache

    Returns:
        Path to the combined embeddings file
    """
    generator = SyntheticMcpArgGenerator()
    return await generator.generate(force=force)


def get_synthetic_embeddings_path() -> Path:
    """Get the path to the combined synthetic embeddings file."""
    return _combined_embeddings_path()


def synthetic_embeddings_exist() -> bool:
    """Check if the combined synthetic embeddings file exists.

    Note: This only checks if the combined file exists, not whether all
    servers have up-to-date embeddings. Run generate_synthetic_embeddings()
    to ensure all embeddings are current.
    """
    return get_synthetic_embeddings_path().exists()


def all_servers_cached() -> bool:
    """Check if all servers in servers.json have cached embeddings."""
    servers_json_path = _synthetic_mcp_dir() / "config" / "servers.json"
    if not servers_json_path.exists():
        return False

    with servers_json_path.open("r", encoding="utf-8") as f:
        servers_config = json.load(f)

    for server_name, server_config in servers_config.items():
        server_hash = _compute_server_hash(server_name, server_config)
        cache_path = _get_server_cache_path(server_name, server_hash)
        if not cache_path.exists():
            return False
    return True


if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv
    asyncio.run(generate_synthetic_embeddings(force=force))
